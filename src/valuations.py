import datetime as dt
import os
from pathlib import Path

import datazimmer as dz
import numpy as np
import pandas as pd
from parquetranger import TableRepo


class ContinentIndex(dz.IndexBase):
    continent_id = str


class ContinentFeatures(dz.TableFeaturesBase):
    name = str


class AreaIndex(dz.IndexBase):
    """Country or region"""

    area_id = str


class CountryIndex(dz.IndexBase):
    country_id = str


class CountryFeatures(dz.TableFeaturesBase):
    name = str
    continent = ContinentIndex
    area = AreaIndex


class CompetitionIndex(dz.IndexBase):
    comp_id = str


class CompetitionFeatures(dz.TableFeaturesBase):
    name = str
    area = AreaIndex
    kind = str


class SeasonIndex(dz.IndexBase):
    season_id = str


class SeasonFeatures(dz.TableFeaturesBase):
    competition = CompetitionIndex
    year_id = str


class PlayerIndex(dz.IndexBase):
    pid = str


class PlayerFeatures(dz.TableFeaturesBase):
    name = str
    preferred_foot = dz.Nullable(str)
    place_of_birth = dz.Nullable(str)
    broad_position = str
    specific_position = dz.Nullable(str)
    date_of_birth = dz.Nullable(dt.datetime)
    national_team_name = dz.Nullable(str)
    national_team_nation = dz.Nullable(str)
    national_app_kind = dz.Nullable(str)
    place_of_birth_country = str
    citizenship_1 = dz.Nullable(str)
    citizenship_2 = dz.Nullable(str)
    full_name = dz.Nullable(str)
    height = dz.Nullable(float)


class TeamIndex(dz.IndexBase):
    team_id = str


class TeamFeatures(dz.TableFeaturesBase):
    name = str
    founded = dz.Nullable(float)
    stadium = dz.Nullable(float)
    members = dz.Nullable(float)
    address = dz.Nullable(str)

    country = CountryIndex

    squad_size = float
    mean_age = float


class MatchIndex(dz.IndexBase):
    match_id = str


class MatchFeatures(dz.TableFeaturesBase):
    score = str
    date = dt.datetime
    home = TeamIndex
    away = TeamIndex
    season = SeasonIndex


class PlayerTransferIndex(dz.IndexBase):
    transfer_id = str


class PlayerTransferFeatures(dz.TableFeaturesBase):
    date = dt.datetime
    is_loan = bool
    is_end_of_loan = bool
    transfer_fee = dz.Nullable(float)

    player = PlayerIndex
    left = TeamIndex
    joined = TeamIndex


class MatchLineupFeatures(dz.TableFeaturesBase):
    side = str
    starter = bool
    country = str
    # representing = CountryIndex
    player_name = str

    player = PlayerIndex
    match = MatchIndex


class PlayerValueIndex(dz.IndexBase):
    mv_id = str


class PlayerValueFeatures(dz.TableFeaturesBase):
    date = dt.datetime
    value = float
    player = PlayerIndex


class TeamRelationFeatures(dz.TableFeaturesBase):
    child = TeamIndex
    parent = TeamIndex


continent_table = dz.ScruTable(ContinentFeatures, ContinentIndex)
country_table = dz.ScruTable(CountryFeatures, CountryIndex)
area_table = dz.ScruTable(index=AreaIndex)
competition_table = dz.ScruTable(CompetitionFeatures, CompetitionIndex)
season_table = dz.ScruTable(SeasonFeatures, SeasonIndex)
player_table = dz.ScruTable(PlayerFeatures, PlayerIndex)
team_table = dz.ScruTable(TeamFeatures, TeamIndex)
match_table = dz.ScruTable(MatchFeatures, MatchIndex)
player_transfer_table = dz.ScruTable(PlayerTransferFeatures, PlayerTransferIndex)
match_lineup_table = dz.ScruTable(MatchLineupFeatures)
player_value_table = dz.ScruTable(PlayerValueFeatures, PlayerValueIndex)
team_relation_table = dz.ScruTable(TeamRelationFeatures)


@dz.register_env_creator
def create_envs(season_sample, match_fraction, value_fraction):

    season_df = season_table.get_full_df().loc[season_sample, :]
    comp_df = competition_table.get_full_df().loc[
        season_df[SeasonFeatures.competition.comp_id].unique()
    ]

    match_df = (
        match_table.get_full_df()
        .loc[lambda df: df[MatchFeatures.season.season_id].isin(season_sample)]
        .sample(frac=match_fraction, random_state=4206969)
    )

    lineup_df = (
        match_lineup_table.get_full_df()
        .loc[lambda df: df[MatchLineupFeatures.match.match_id].isin(match_df.index)]
        .reset_index(drop=True)
    )

    player_df = player_table.get_full_df().loc[
        lineup_df[MatchLineupFeatures.player.pid].unique(), :
    ]

    # TODO: this seems like composite type
    transfer_df, mv_df = [
        trepo.get_full_df()
        .loc[
            lambda df: df[PlayerValueFeatures.player.pid].isin(player_df.index)
            & (df[PlayerValueFeatures.date] < match_df[MatchFeatures.date].max())
            & (
                df[PlayerValueFeatures.date]
                > match_df[MatchFeatures.date].min() - dt.timedelta(days=365)
            )
        ]
        .sample(frac=value_fraction, random_state=2323)
        for trepo in [player_transfer_table, player_value_table]
    ]

    _teams1 = match_df.loc[:, [MatchFeatures.home.team_id, MatchFeatures.away.team_id]]
    _teams2 = transfer_df[
        [
            PlayerTransferFeatures.joined.team_id,
            PlayerTransferFeatures.left.team_id,
        ]
    ]
    all_teams = list(
        set(sum([_tdf.unstack().tolist() for _tdf in [_teams1, _teams2]], []))
    )

    team_df = (
        team_table.get_full_df().reindex(all_teams).dropna(subset=[TeamFeatures.name])
    )

    team_rel_df = (
        team_relation_table.get_full_df()
        .loc[
            lambda df: df[TeamRelationFeatures.parent.team_id].isin(team_df.index)
            & df[TeamRelationFeatures.child.team_id].isin(team_df.index)
        ]
        .reset_index(drop=True)
    )

    country_df = country_table.get_full_df()
    continent_df = continent_table.get_full_df()
    area_df = area_table.get_full_df()

    dz.dump_dfs_to_tables(
        [
            (continent_df, continent_table),
            (country_df, country_table),
            (area_df, area_table),
            (comp_df, competition_table),
            (season_df, season_table),
            (player_df, player_table),
            (team_df, team_table),
            (match_df, match_table),
            (transfer_df, player_transfer_table),
            (lineup_df, match_lineup_table),
            (mv_df, player_value_table),
            (team_rel_df, team_relation_table),
        ],
    )


@dz.register_data_loader
def load():
    external_dir = Path(os.environ["FOOTBALL_VALUE_DATA_ROOT"])

    ext_countries_table = TableRepo(external_dir / "countries")
    ext_season_info_table = TableRepo(external_dir / "season_info")
    ext_player_info_table = TableRepo(external_dir / "player_info")
    ext_match_info_table = TableRepo(external_dir / "match_info")
    ext_player_transfers_table = TableRepo(external_dir / "player_transfers")
    ext_team_info_table = TableRepo(external_dir / "team_info")
    ext_match_lineups_table = TableRepo(external_dir / "match_lineups")
    ext_player_values_table = TableRepo(external_dir / "player_values")
    ext_team_relations_table = TableRepo(external_dir / "team_relations")

    country_df = ext_countries_table.get_full_df()
    season_df = ext_season_info_table.get_full_df()
    team_df = ext_team_info_table.get_full_df().rename(
        columns={"country": TeamFeatures.country.country_id}
    )
    player_df = (
        ext_player_info_table.get_full_df()
        .reset_index()
        .rename(
            columns={
                "tm_player_id": PlayerIndex.pid,
                "dob": PlayerFeatures.date_of_birth,
                "citizenship-0": PlayerFeatures.citizenship_1,
                "citizenship-1": PlayerFeatures.citizenship_2,
            }
        )
    )
    match_df = (
        ext_match_info_table.get_full_df()
        .rename(
            columns={
                "home-tm_id": MatchFeatures.home.team_id,
                "away-tm_id": MatchFeatures.away.team_id,
            }
        )
        .assign(
            **{
                MatchFeatures.season.season_id: lambda df: df["comp_id"]
                + "-"
                + df["season_id"]
            }
        )
    )

    continent_table.replace_all(
        country_df.rename(columns={"continent": ContinentIndex.continent_id})
        .groupby(ContinentIndex.continent_id)
        .count()
        .assign(name=lambda df: pd.Series({"amerika": "America", "europa": "Europe"}))
    )
    country_table.replace_all(
        country_df.rename(
            columns={"continent": CountryFeatures.continent.continent_id}
        ).assign(**{CountryFeatures.area.area_id: lambda df: df.index})
    )

    area_table.replace_all(
        season_df.groupby("country")
        .count()
        .pipe(
            lambda df: pd.DataFrame(
                index=pd.Index(df.index.union(country_df.index), name=AreaIndex.area_id)
            )
        )
    )

    competition_table.replace_all(
        season_df.groupby(CompetitionIndex.comp_id)
        .first()
        .rename(columns={"country": CompetitionFeatures.area.area_id})
        .assign(
            **{
                CompetitionFeatures.kind: lambda df: np.where(
                    df["base"] == "pokalwettbewerb", "cup", "league"
                )
            }
        )
    )

    season_table.replace_all(
        season_df.reset_index().rename(
            columns={
                "uid": SeasonIndex.season_id,
                "season_id": SeasonFeatures.year_id,
                "comp_id": SeasonFeatures.competition.comp_id,
            }
        )
    )

    lup_df = (
        ext_match_lineups_table.get_full_df()
        .assign(starter=lambda df: df["starter"] == "starter")
        .rename(
            columns={
                "name": MatchLineupFeatures.player_name,
                "tm_id": MatchLineupFeatures.player.pid,
                "match_id": MatchLineupFeatures.match.match_id,
            }
        )
    )

    transfer_df = ext_player_transfers_table.get_full_df().rename(
        columns={
            "tm_player_id": PlayerTransferFeatures.player.pid,
            "left": PlayerTransferFeatures.left.team_id,
            "joined": PlayerTransferFeatures.joined.team_id,
        }
    )

    mv_df = ext_player_values_table.get_full_df().rename(
        columns={"tm_player_id": PlayerValueFeatures.player.pid}
    )

    team_rel_df = (
        ext_team_relations_table.get_full_df()
        .reset_index()
        .rename(
            columns={
                "child_team_id": TeamRelationFeatures.child.team_id,
                "parent_team_id": TeamRelationFeatures.parent.team_id,
            }
        )
        .dropna(how="any")
    )

    dz.dump_dfs_to_tables(
        [
            (player_df, player_table),
            (team_df, team_table),
            (match_df, match_table),
            (transfer_df, player_transfer_table),
            (lup_df, match_lineup_table),
            (mv_df, player_value_table),
            (team_rel_df, team_relation_table),
        ],
    )
