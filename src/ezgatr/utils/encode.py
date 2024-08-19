import torch

from ezgatr.primitives.bilinear import geometric_product


class Point:
    """Interface for 3D Euclidean point to and from PGA representation.

    In PGA [1]_, a point is represented as the intersection of three planes,
    and represented with tri-vectors `e_{0ij}`.

    TODO: WE MAY WANT TO ADD MORE EXPLANATION OF WHY THERE ARE SIGN FLIPS IN THE
    ENCODING AND DECODING FUNCTIONS.

    Reference
    ---------
    .. [1] `"A Guided Tour to the Plane-Based Geometric Algebra PGA", Leo Dorst,
            <https://geometricalgebra.org/downloads/PGA4CS.pdf>`_
    """

    @staticmethod
    def encode(points):
        """Encode 3D Euclidean points to PGA.

        Parameters
        ----------
        points : torch.Tensor
            3D Euclidean points with shape (..., 3).

        Returns
        -------
        torch.Tensor
            PGA representation of the points with shape (..., 16).
        """
        mvs = torch.zeros(
            *points.shape[:-1], 16, dtype=points.dtype, device=points.device
        )

        mvs[..., 14] = 1.0
        mvs[..., 13] = -points[..., 0]
        mvs[..., 12] = points[..., 1]
        mvs[..., 11] = -points[..., 2]

        return mvs

    @staticmethod
    def decode(mvs):
        """ """
        pass


class Plane:
    """Interface for oriented plane to and from PGA

    Indices 2, 3, 4 are vectors with basis `e_1, e_2, e_3`. Vectors
    with basis `e_1, e_2, e_3` are planes, which are represented
    as direction `(e_1, e_2, e_3)` and distance `(e_0)` in PGA.
    """

    @staticmethod
    def encode(normals, distance):
        """Encode oriented planes to PGA.

        Parameters
        ----------
        normals : torch.Tensor
            Normal vectors of the planes with shape (..., 3).
        distance : torch.Tensor
            One position on the planes with shape (..., 3),
            or the distance to the origin (scalar).

        Returns
        -------
        torch.Tensor
            PGA representation of the planes with shape (..., 16).
        """
        mvs = torch.zeros(
            *normals.shape[:-1], 16, dtype=normals.dtype, device=normals.device
        )

        mvs[..., 2:5] = normals[..., :]

        # Check whether distance is a scalar or a vector
        if distance.dim() == 1:
            mvs[..., 1] = distance
            return mvs

        translation = Translation.encode(distance)
        inverse_translation = Translation.encode(-distance)

        # From page 55 of PGA4CS,
        #   "In the sandwiching with this element T_t (the translator),
        #   any element translates over t"
        mvs = geometric_product(
            geometric_product(translation, mvs), inverse_translation
        )

        return mvs

    @staticmethod
    def decode(mvs):
        # TODO: Need to figure out a way to extract both direction and distance from the multivector
        pass


class Scalar:
    """Interface for scalar to and from PGA representation."""

    @staticmethod
    def encode(scalars):
        pad = torch.zeros(
            *scalars.shape[:-1], 15, dtype=scalars.dtype, device=scalars.device
        )
        return torch.cat([scalars, pad], dim=-1)

    @staticmethod
    def decode(mvs):
        return mvs[..., [0]]


class Pseudoscalar:
    """Interface for pseudoscalar to and from PGA representation."""

    @staticmethod
    def encode(pseudoscalars):
        pad = torch.zeros(
            *pseudoscalars.shape[:-1],
            15,
            dtype=pseudoscalars.dtype,
            device=pseudoscalars.device
        )
        return torch.cat([pad, pseudoscalars], dim=-1)

    @staticmethod
    def decode(mvs):
        return mvs[..., [15]]


class Reflection:
    """Interface for reflection to and from PGA representation.

    In PGA [1]_, a plane can serve as a reflection operator. Therefore, a reflection
    is encoded exactly as a plane.

    Reference
    ---------
    .. [1] `"A Guided Tour to the Plane-Based Geometric Algebra PGA", Leo Dorst,
            <https://geometricalgebra.org/downloads/PGA4CS.pdf>`_
    """

    @staticmethod
    def encode(normals, positions):
        return Plane.encode(normals, positions)

    @staticmethod
    def decode(mvs):
        return Plane.decode(mvs)


class Rotation:
    """Interface for rotation to and from PGA"""

    @staticmethod
    def encode(quaternions):
        pass

    @staticmethod
    def decode(mvs):
        pass


class Translation:
    """Interface for translation to and from PGA.

    The equation can be found on page 55 of PGA4CS (equation 82) where translator
    is defined as,

    ```
    T_t = 1 + e_0 t / 2
    ```

    where `t` is define as `2(\delta_2 - \delta_1)n`, `e_0` corresponds to `e`
    in the textbook as it is the basis for distance, and `n` is a unit normal
    vector. Therefore, the outputted multivector should be bivector with basis
    `e_{0i}`, which corresponds to indices 5, 6, 7.
    """

    @staticmethod
    def encode(delta):
        """Encode translation to PGA.

        Parameters
        ----------
        delta : torch.Tensor
            Translation amount with shape (..., 3).

        Returns
        -------
        torch.Tensor
            PGA representation of the translator with shape (..., 16).
        """
        mvs = torch.zeros(*delta.shape[:-1], 16, dtype=delta.dtype, device=delta.device)

        mvs[..., 0] = 1.0
        mvs[..., 5:8] = -0.5 * delta[..., :]

        return mvs

    @staticmethod
    def decode(mvs, divide_by_embedding_dim=False, threshold: float = 1e-3):
        """Decode translation from PGA.

        Parameters
        ----------
        mvs : torch.Tensor
            Multivector with shape (..., 16).
        divide_by_embedding_dim : bool
            Whether to divide by the embedding dim. Proper PGA etiquette would have us
            do this, but it may not be good for NN training.
        threshold : float
            Minimum value of the additional, un-physical component. Necessary to avoid
            exploding values or NaNs when this un-physical component of the homogeneous
            coordinates becomes small.

        Returns
        -------
        torch.Tensor
            3D components of the translation vector.
        """
        delta = -2.0 * mvs[..., 5:8]

        if divide_by_embedding_dim:

            # The scalar part of a translation versor is always equal to 1 (page 85 of PGA4CS)
            embedding_dim = mvs[..., [0]]

            # Avoid division by zero, exploding values or NaNs
            embedding_dim = torch.where(
                torch.abs(embedding_dim) > threshold, embedding_dim, threshold
            )
            delta = delta / embedding_dim

        return delta
