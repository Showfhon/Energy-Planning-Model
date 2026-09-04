"""Tests for profile synthesis and representative-day reduction."""

import csv
import os
import tempfile
import unittest

from energyplan.timeseries import (
    DAYS_PER_YEAR,
    HOURS_PER_YEAR,
    aggregate_hours,
    cluster_days,
    load_profiles_csv,
    synthesise_profiles,
)


class TestSyntheticProfiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profiles = synthesise_profiles(seed=1234)

    def test_length_and_keys(self):
        for name in ("demand", "solar", "wind_onshore", "wind_offshore"):
            self.assertEqual(len(self.profiles[name]), HOURS_PER_YEAR, name)

    def test_demand_shape_has_unit_mean(self):
        demand = self.profiles["demand"]
        self.assertAlmostEqual(sum(demand) / len(demand), 1.0, places=9)

    def test_capacity_factors_stay_in_range(self):
        for name in ("solar", "wind_onshore", "wind_offshore"):
            series = self.profiles[name]
            self.assertGreaterEqual(min(series), 0.0, name)
            self.assertLessEqual(max(series), 1.0, name)

    def test_solar_is_zero_at_night_and_positive_at_noon(self):
        solar = self.profiles["solar"]
        midnights = [solar[day * 24 + 0] for day in range(DAYS_PER_YEAR)]
        noons = [solar[day * 24 + 12] for day in range(DAYS_PER_YEAR)]
        self.assertEqual(max(midnights), 0.0)
        self.assertGreater(sum(noons) / len(noons), 0.3)

    def test_annual_capacity_factors_are_plausible(self):
        def mean(name):
            return sum(self.profiles[name]) / HOURS_PER_YEAR

        self.assertTrue(0.10 < mean("solar") < 0.35)
        self.assertTrue(0.15 < mean("wind_onshore") < 0.45)
        self.assertTrue(0.30 < mean("wind_offshore") < 0.60)

    def test_deterministic(self):
        again = synthesise_profiles(seed=1234)
        self.assertEqual(again["demand"][:50], self.profiles["demand"][:50])


class TestAggregation(unittest.TestCase):
    def test_averages_blocks(self):
        day = list(range(24))
        self.assertEqual(aggregate_hours(day, 24), day)
        self.assertEqual(len(aggregate_hours(day, 4)), 4)
        self.assertAlmostEqual(aggregate_hours(day, 4)[0], sum(range(6)) / 6)

    def test_rejects_non_divisor(self):
        with self.assertRaises(ValueError):
            aggregate_hours(list(range(24)), 7)


class TestClustering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profiles = synthesise_profiles(seed=99)

    def test_weights_reproduce_the_year(self):
        for n_days in (1, 4, 8, 16):
            rep = cluster_days(self.profiles, n_days=n_days, hours_per_day=24, seed=7)
            self.assertAlmostEqual(sum(rep.weights), DAYS_PER_YEAR, places=6)
            self.assertAlmostEqual(rep.total_hours(), HOURS_PER_YEAR, places=6)

    def test_hours_per_day_scales_the_step(self):
        rep = cluster_days(self.profiles, n_days=4, hours_per_day=6, seed=7)
        self.assertEqual(rep.hours_per_step, 4.0)
        self.assertAlmostEqual(rep.total_hours(), HOURS_PER_YEAR, places=6)

    def test_peak_day_is_retained(self):
        demand = self.profiles["demand"]
        daily_peak = [max(demand[d * 24:(d + 1) * 24]) for d in range(DAYS_PER_YEAR)]
        expected = max(range(DAYS_PER_YEAR), key=lambda d: daily_peak[d])
        rep = cluster_days(
            self.profiles, n_days=6, hours_per_day=24, seed=7, net_load=demand
        )
        self.assertEqual(rep.day_indices[rep.peak_day_position], expected)

    def test_medoids_are_distinct_real_days(self):
        rep = cluster_days(self.profiles, n_days=8, hours_per_day=24, seed=3)
        self.assertEqual(len(set(rep.day_indices)), 8)
        for day in rep.day_indices:
            self.assertTrue(0 <= day < DAYS_PER_YEAR)

    def test_slice_profile_matches_the_source_day(self):
        rep = cluster_days(self.profiles, n_days=3, hours_per_day=24, seed=3)
        sliced = rep.slice_profile(self.profiles["solar"])
        first = rep.day_indices[0]
        self.assertEqual(sliced[0], self.profiles["solar"][first * 24:first * 24 + 24])

    def test_deterministic_for_a_given_seed(self):
        a = cluster_days(self.profiles, n_days=5, seed=11)
        b = cluster_days(self.profiles, n_days=5, seed=11)
        self.assertEqual(a.day_indices, b.day_indices)


class TestCsvProfiles(unittest.TestCase):
    def test_round_trip_and_demand_normalisation(self):
        profiles = synthesise_profiles(seed=5)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "profiles.csv")
            with open(path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["demand", "solar"])
                for hour in range(HOURS_PER_YEAR):
                    writer.writerow([profiles["demand"][hour] * 7.0,
                                     profiles["solar"][hour]])
            loaded = load_profiles_csv(path)
        self.assertAlmostEqual(sum(loaded["demand"]) / HOURS_PER_YEAR, 1.0, places=9)
        self.assertAlmostEqual(loaded["solar"][12], profiles["solar"][12], places=9)

    def test_wrong_length_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "short.csv")
            with open(path, "w", encoding="utf-8", newline="") as handle:
                handle.write("demand\n1.0\n2.0\n")
            with self.assertRaises(ValueError):
                load_profiles_csv(path)


if __name__ == "__main__":
    unittest.main()
