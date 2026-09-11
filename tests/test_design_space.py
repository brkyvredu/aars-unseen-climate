from classguard_v2.design_space import (
    CITIES,
    DesignSpaceConfig,
    expand_across_cities,
    generate_base_designs,
    validate_paired_design_space,
)


def test_paired_designs_are_exactly_repeated_across_cities():
    base = generate_base_designs(DesignSpaceConfig(n_base_designs=24, seed=7))
    paired = expand_across_cities(base)
    report = validate_paired_design_space(paired)
    assert report["valid"]
    assert len(paired) == 24 * len(CITIES)
    assert report["incomplete_base_designs"] == 0
    assert report["mismatched_design_columns"] == []
