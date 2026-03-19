"""Data I/O helpers."""

from __future__ import annotations

import re
import warnings
from pathlib import Path

_CREDENTIAL_WARNING = "credentials were not supplied. open data access only"


def _default_spadl_filename(competition_name: str, season_name: str) -> str:
    competition_slug = re.sub(r"[^a-z0-9]+", "", competition_name.lower())
    season_slug = re.sub(r"[^a-z0-9]+", "", str(season_name).lower())
    return f"spadl-{competition_slug}{season_slug}"


def save_spadl_h5(
    competition_name: str,
    season_name: str,
    *,
    datafolder: str | Path = "data",
    filename: str | None = None,
    show_progress: bool = True,
) -> Path:
    """
    Download StatsBomb open data and store SPADL tables in an HDF5 file.

    The generated file layout matches `experiment/01-download-data.ipynb`.

    Parameters
    ----------
    competition_name : str
        Competition display name in StatsBomb (e.g. ``"FIFA World Cup"``).
    season_name : str
        Season display name in StatsBomb (e.g. ``"2018"``).
    datafolder : str | Path, default="data"
        Output directory where the HDF5 file is written.
    filename : str | None, default=None
        HDF5 filename stem (without extension). If omitted, a name is built
        from competition and season.
    show_progress : bool, default=True
        Show tqdm progress bar while loading per-game data.

    Returns
    -------
    Path
        Absolute path to the generated HDF5 file.

    Raises
    ------
    ValueError
        If the competition/season pair is not available, or no games exist.
    """
    import pandas as pd
    import tqdm
    from socceraction.data.statsbomb import StatsBombLoader
    import socceraction.spadl as spadl

    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message=_CREDENTIAL_WARNING)
        sbl = StatsBombLoader(getter="remote", creds={"user": None, "passwd": None})
        competitions = sbl.competitions()

    selected_competitions = competitions[
        (competitions.competition_name == competition_name)
        & (competitions.season_name.astype(str) == str(season_name))
    ]
    if selected_competitions.empty:
        valid_seasons = sorted(
            competitions.loc[
                competitions.competition_name == competition_name, "season_name"
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        if valid_seasons:
            raise ValueError(
                f"No data found for competition={competition_name!r}, "
                f"season={season_name!r}. Available seasons: {valid_seasons}."
            )
        raise ValueError(
            f"Competition {competition_name!r} was not found in StatsBomb open data."
        )

    games = pd.concat(
        [
            sbl.games(row.competition_id, row.season_id)
            for row in selected_competitions.itertuples()
        ],
        ignore_index=True,
    )
    if games.empty:
        raise ValueError(
            "No games found for competition "
            f"{competition_name!r} in season {season_name!r}."
        )

    game_rows = list(games.itertuples())
    if show_progress:
        game_rows = tqdm.tqdm(game_rows, desc="Loading game data")

    teams, players = [], []
    actions: dict[int, pd.DataFrame] = {}
    for game in game_rows:
        teams.append(sbl.teams(game.game_id))
        players.append(sbl.players(game.game_id))
        events = sbl.events(game.game_id)
        actions[game.game_id] = spadl.statsbomb.convert_to_actions(
            events,
            home_team_id=game.home_team_id,
            xy_fidelity_version=1,
            shot_fidelity_version=1,
        )

    teams_df = pd.concat(teams, ignore_index=True).drop_duplicates(subset="team_id")
    players_df = pd.concat(players, ignore_index=True)

    output_dir = Path(datafolder)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = filename or _default_spadl_filename(competition_name, season_name)
    output_path = output_dir / f"{output_stem}.h5"

    with pd.HDFStore(output_path) as spadlstore:
        spadlstore["competitions"] = selected_competitions
        spadlstore["games"] = games
        spadlstore["teams"] = teams_df
        spadlstore["players"] = players_df[
            ["player_id", "player_name", "nickname"]
        ].drop_duplicates(subset="player_id")
        spadlstore["player_games"] = players_df[
            [
                "player_id",
                "game_id",
                "team_id",
                "is_starter",
                "starting_position_id",
                "starting_position_name",
                "minutes_played",
            ]
        ]
        for game_id, game_actions in actions.items():
            spadlstore[f"actions/game_{game_id}"] = game_actions

    return output_path.resolve()
