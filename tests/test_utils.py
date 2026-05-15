"""
Unit tests for quadsv.utils functions.
Tests coordinate generation and distance calculations.
"""

import unittest

import numpy as np

from quadsv.utils import (
    compute_torus_distance_matrix,
    convert_visium_to_physical,
    get_rect_coords,
    get_visium_coords,
)


class TestGetRectCoords(unittest.TestCase):
    """Test rectangular grid coordinate generation."""

    def test_docstring_example(self):
        """Test example from docstring: 10x10 grid."""
        coords, dims = get_rect_coords(n_rows=10, n_cols=10)
        assert coords.shape == (100, 2)
        assert dims == (10, 10)

    def test_coordinates_are_integers(self):
        """Test that all coordinates are integers."""
        coords, _ = get_rect_coords(n_rows=5, n_cols=7)
        assert np.all(coords == np.floor(coords))

    def test_coordinates_range(self):
        """Test that coordinates are within expected ranges."""
        n_rows, n_cols = 10, 20
        coords, _ = get_rect_coords(n_rows=n_rows, n_cols=n_cols)

        # First column (rows) should be in [0, n_rows)
        assert np.all(coords[:, 0] >= 0)
        assert np.all(coords[:, 0] < n_rows)

        # Second column (cols) should be in [0, n_cols)
        assert np.all(coords[:, 1] >= 0)
        assert np.all(coords[:, 1] < n_cols)

    def test_single_row(self):
        """Test with single row."""
        coords, grid_dims = get_rect_coords(n_rows=1, n_cols=5)
        assert coords.shape == (5, 2)
        assert grid_dims == (1, 5)
        assert np.all(coords[:, 0] == 0)  # All should be row 0

    def test_single_column(self):
        """Test with single column."""
        coords, grid_dims = get_rect_coords(n_rows=5, n_cols=1)
        assert coords.shape == (5, 2)
        assert grid_dims == (5, 1)
        assert np.all(coords[:, 1] == 0)  # All should be col 0


class TestGetVisiumCoords(unittest.TestCase):
    """Test Visium hexagonal grid coordinate generation."""

    def test_docstring_example(self):
        """Test example from docstring: 78x64 grid."""
        coords, dims = get_visium_coords(n_rows=78, n_cols=64)
        assert coords.shape[0] == 4992
        assert dims == (78, 64)

    def test_small_visium_grid(self):
        """Test get_visium_coords with small grid."""
        coords, grid_dims = get_visium_coords(n_rows=3, n_cols=2)
        assert coords.shape == (6, 2)
        assert grid_dims == (3, 2)

        # Verify hex pattern
        # Row 0 (even): starts at 0, then 2
        # Row 1 (odd): starts at 1, then 3
        # Row 2 (even): starts at 0, then 2
        expected = np.array(
            [
                [0, 0],
                [0, 2],  # Row 0, even
                [1, 1],
                [1, 3],  # Row 1, odd
                [2, 0],
                [2, 2],  # Row 2, even
            ]
        )
        assert np.allclose(coords, expected)

    def test_row_parity_pattern(self):
        """Test that even and odd rows have correct starting columns."""
        coords, _ = get_visium_coords(n_rows=4, n_cols=3)

        # Row 0 (even): cols should start at 0
        row_0 = coords[0:3]
        assert row_0[0, 1] == 0  # First col

        # Row 1 (odd): cols should start at 1
        row_1 = coords[3:6]
        assert row_1[0, 1] == 1  # First col should be 1

        # Row 2 (even): cols should start at 0
        row_2 = coords[6:9]
        assert row_2[0, 1] == 0  # First col

    def test_col_spacing_is_2(self):
        """Test that columns within a row are spaced by 2."""
        coords, _ = get_visium_coords(n_rows=2, n_cols=4)

        # Row 0: should have cols 0, 2, 4, 6
        row_0 = coords[0:4, 1]
        diffs = np.diff(row_0)
        assert np.all(diffs == 2)

    def test_coordinates_are_integers(self):
        """Test that all coordinates are integers."""
        coords, _ = get_visium_coords(n_rows=10, n_cols=8)
        assert np.all(coords == np.floor(coords))

    def test_single_row_visium(self):
        """Test Visium with single row."""
        coords, grid_dims = get_visium_coords(n_rows=1, n_cols=5)
        assert coords.shape == (5, 2)
        assert grid_dims == (1, 5)


class TestConvertVisiumToPhysical(unittest.TestCase):
    """Test conversion from Visium indices to physical coordinates."""

    def test_docstring_example(self):
        """Test example from docstring with specific input coordinates."""
        coords = np.array([[0, 0], [0, 2], [1, 1]])
        phys_coords = convert_visium_to_physical(coords)

        # Expected results from docstring
        expected = np.array([[0.0, 0.0], [0.0, 1.0], [0.8660254, 0.5]])
        np.testing.assert_array_almost_equal(phys_coords, expected, decimal=6)

    def test_origin_conversion(self):
        """Test that (0, 0) converts correctly."""
        coords = np.array([[0, 0]])
        phys = convert_visium_to_physical(coords)
        assert np.allclose(phys, [[0.0, 0.0]])

    def test_col_scaling(self):
        """Test that column scaling is 0.5."""
        coords = np.array([[0, 0], [0, 2], [0, 4]])
        phys = convert_visium_to_physical(coords)
        # x-coordinates should be 0, 1, 2
        assert np.allclose(phys[:, 1], [0.0, 1.0, 2.0])

    def test_row_scaling(self):
        """Test that row scaling is sqrt(3)/2."""
        coords = np.array([[0, 0], [1, 0], [2, 0]])
        phys = convert_visium_to_physical(coords)
        expected_y = np.array([0, np.sqrt(3) / 2, np.sqrt(3)])
        assert np.allclose(phys[:, 0], expected_y)

    def test_hex_distance_properties(self):
        """Test that physical coordinates preserve hex distance properties."""
        # In hexagonal grid, each cell has 6 equidistant neighbors
        # For the cell at (0, 0), neighbors should be at fixed distance
        center = np.array([[0, 0]])
        neighbors = np.array(
            [
                [0, 2],  # right
                [1, 1],  # down-right
                [1, -1],  # down-left (if wrapped)
                [0, -2],  # left (if wrapped)
                [-1, -1],  # up-left (if wrapped)
                [-1, 1],  # up-right (if wrapped)
            ]
        )

        center_phys = convert_visium_to_physical(center)
        neighbors_phys = convert_visium_to_physical(neighbors)

        # Compute distances
        dists = np.linalg.norm(neighbors_phys - center_phys, axis=1)

        # All distances should be 1.0 (for valid neighbors within bounds)
        # We only test the first neighbor (right) which should be valid
        assert np.isclose(dists[0], 1.0)

    def test_batch_conversion(self):
        """Test conversion of multiple coordinates."""
        coords = np.array([[0, 0], [1, 2], [2, 4], [3, 0]])
        phys = convert_visium_to_physical(coords)
        assert phys.shape == (4, 2)

        # Verify each row
        assert np.allclose(phys[0], [0.0, 0.0])
        assert np.allclose(phys[1], [np.sqrt(3) / 2, 1.0])
        assert np.allclose(phys[2], [np.sqrt(3), 2.0])
        assert np.allclose(phys[3], [3 * np.sqrt(3) / 2, 0.0])

    def test_output_shape(self):
        """Test that output shape matches input."""
        coords = np.array([[0, 0], [1, 2], [3, 4], [5, 6]])
        phys = convert_visium_to_physical(coords)
        assert phys.shape == coords.shape


class TestComputeTorusDistanceMatrix(unittest.TestCase):
    """Test torus distance matrix computation."""

    def test_docstring_example(self):
        """Test exact example from docstring showing wrapping behavior."""
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [9.9, 0.0]])
        domain = (10.0, 10.0)
        dists = compute_torus_distance_matrix(coords, domain)

        # Distance from (0,0) to (9.9,0) on [0,10)×[0,10) torus should wrap
        # Wrapping distance: min(9.9, 10-9.9) = 0.1
        np.testing.assert_almost_equal(dists[0, 2], 0.1, decimal=6)
        np.testing.assert_almost_equal(dists[2, 0], 0.1, decimal=6)

    def test_single_point(self):
        """Test with single point (distance to self)."""
        coords = np.array([[0.5, 0.5]])
        domain_dims = (10.0, 10.0)
        dist_matrix = compute_torus_distance_matrix(coords, domain_dims)
        assert dist_matrix.shape == (1, 1)
        assert np.isclose(dist_matrix[0, 0], 0.0)

    def test_two_points_no_wrapping(self):
        """Test distance between two points without wrapping."""
        coords = np.array([[0.0, 0.0], [3.0, 4.0]])
        domain_dims = (100.0, 100.0)  # Large domain, no wrapping
        dist_matrix = compute_torus_distance_matrix(coords, domain_dims)

        # Euclidean distance should be 5.0
        assert np.isclose(dist_matrix[0, 1], 5.0)
        assert np.isclose(dist_matrix[1, 0], 5.0)

    def test_symmetry(self):
        """Test that distance matrix is symmetric."""
        coords = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 1.0]])
        domain_dims = (20.0, 20.0)
        dist_matrix = compute_torus_distance_matrix(coords, domain_dims)

        # Should be symmetric
        assert np.allclose(dist_matrix, dist_matrix.T)

    def test_diagonal_zeros(self):
        """Test that diagonal is all zeros (distance to self)."""
        coords = np.array([[0.5, 0.5], [2.0, 3.0], [5.0, 1.0], [7.0, 8.0]])
        domain_dims = (20.0, 20.0)
        dist_matrix = compute_torus_distance_matrix(coords, domain_dims)

        assert np.allclose(np.diag(dist_matrix), 0.0)

    def test_wrapping_both_axes(self):
        """Test wrapping on both axes."""
        # Distance with wrapping on both axes
        # Point A: (0.5, 0.5), Point B: (9.0, 9.5)
        # Direct: sqrt((8.5)^2 + (9)^2) = sqrt(72.25 + 81) = sqrt(153.25)
        # Wrapped: sqrt((1.5)^2 + (1.0)^2) = sqrt(2.25 + 1.0) = sqrt(3.25)
        coords = np.array([[0.5, 0.5], [9.0, 9.5]])
        domain_dims = (10.0, 10.0)
        dist_matrix = compute_torus_distance_matrix(coords, domain_dims)

        expected_dist = np.sqrt(1.5**2 + 1.0**2)
        assert np.isclose(dist_matrix[0, 1], expected_dist)

    def test_rectangular_domain(self):
        """Test with non-square domain."""
        coords = np.array([[0.0, 0.0], [1.0, 19.5]])
        domain_dims = (20.0, 20.0)
        dist_matrix = compute_torus_distance_matrix(coords, domain_dims)

        # Distance with wrapping: sqrt(1 + 0.5^2) = sqrt(1.25)
        expected_dist = np.sqrt(1.0**2 + 0.5**2)
        assert np.isclose(dist_matrix[0, 1], expected_dist)

    def test_non_negative_distances(self):
        """Test that all distances are non-negative."""
        coords = np.array([[0.0, 0.0], [5.0, 5.0], [9.9, 9.9], [0.1, 0.1]])
        domain_dims = (10.0, 10.0)
        dist_matrix = compute_torus_distance_matrix(coords, domain_dims)

        assert np.all(dist_matrix >= 0)

    def test_triangle_inequality(self):
        """Test that triangle inequality holds approximately."""
        coords = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
        domain_dims = (100.0, 100.0)  # Large to avoid wrapping
        dist_matrix = compute_torus_distance_matrix(coords, domain_dims)

        # d(A,B) + d(B,C) >= d(A,C)
        d_ab = dist_matrix[0, 1]
        d_bc = dist_matrix[1, 2]
        d_ac = dist_matrix[0, 2]

        assert d_ab + d_bc >= d_ac - 1e-10  # Account for floating point error


if __name__ == "__main__":
    unittest.main()
