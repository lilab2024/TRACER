def trim(frame, first_step=2000):
    """Discard Time 0..1999; keep original Time values for later alignment."""
    result = frame.loc[frame.Time >= first_step].copy().reset_index(drop=True)
    if result.empty or result.Time.duplicated().any():
        raise ValueError('Empty trajectory or duplicate Time')
    return result
