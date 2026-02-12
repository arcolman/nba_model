def moneyline_to_prob(ml: float) -> float:
    """
    American moneyline -> implied probability (no-vig not handled here).
    -150 -> 0.60
    +130 -> 0.435
    """
    ml = float(ml)
    if ml < 0:
        return (-ml) / ((-ml) + 100.0)
    else:
        return 100.0 / (ml + 100.0)
