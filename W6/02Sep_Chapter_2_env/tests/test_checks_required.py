from pvch2.checks.sections import check_3_3_2, check_3_3_4, check_3_3_5, check_3_3_6, check_3_4


def test_required_sections_pass():
    for fn in (check_3_3_2, check_3_3_4, check_3_3_5, check_3_3_6, check_3_4):
        result = fn()
        failed = [i.name for i in result.items if not i.ok]
        assert result.passed, f"{result.section} failed: {failed}"
