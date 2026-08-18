"""
FantAssistant — Funzioni di upsert per Delta Lake
Enforce dei vincoli UNIQUE tramite MERGE INTO.

Ogni funzione accetta un DataFrame Spark incoming e fa merge sulla/e
colonna/e che costituiscono la chiave naturale (ex vincolo UNIQUE).

Usage:
    from upsert_utils import upsert_squadre, upsert_voti_giornata, ...
"""
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from delta.tables import DeltaTable

CATALOG = "platform"
SCHEMA = "fantassistant"
NS = f"`{CATALOG}`.`{SCHEMA}`"


def _get_spark() -> SparkSession:
    return SparkSession.builder.getOrCreate()


# --------------------------------------------------------------------------
# giocatori — UNIQUE(fanta_id), UNIQUE(nome, squadra)
# --------------------------------------------------------------------------
def upsert_giocatori(incoming: DataFrame) -> None:
    """Upsert giocatori su fanta_id. Aggiorna se esiste, inserisce se nuovo."""
    spark = _get_spark()
    dt = DeltaTable.forName(spark, f"{CATALOG}.{SCHEMA}.giocatori")
    (
        dt.alias("t")
        .merge(
            incoming.alias("s"),
            "t.fanta_id = s.fanta_id"
        )
        .whenMatchedUpdate(set={
            "nome": "s.nome",
            "ruolo": "s.ruolo",
            "squadra": "s.squadra",
            "quotazione_iniziale": "s.quotazione_iniziale",
            "quotazione_attuale": "s.quotazione_attuale",
            "fvm": "s.fvm",
            "aggiornato_il": F.current_timestamp(),
        })
        .whenNotMatchedInsert(values={
            "fanta_id": "s.fanta_id",
            "nome": "s.nome",
            "ruolo": "s.ruolo",
            "squadra": "s.squadra",
            "quotazione_iniziale": "s.quotazione_iniziale",
            "quotazione_attuale": "s.quotazione_attuale",
            "fvm": "s.fvm",
            "creato_il": F.current_timestamp(),
            "aggiornato_il": F.current_timestamp(),
        })
        .execute()
    )


# --------------------------------------------------------------------------
# statistiche_storiche — UNIQUE(giocatore_id, stagione)
# --------------------------------------------------------------------------
def upsert_statistiche_storiche(incoming: DataFrame) -> None:
    """Upsert statistiche su (giocatore_id, stagione)."""
    spark = _get_spark()
    dt = DeltaTable.forName(spark, f"{CATALOG}.{SCHEMA}.statistiche_storiche")
    (
        dt.alias("t")
        .merge(
            incoming.alias("s"),
            "t.giocatore_id = s.giocatore_id AND t.stagione = s.stagione"
        )
        .whenMatchedUpdate(set={
            "giocatore_id": "s.giocatore_id",
            "stagione": "s.stagione",
            "presenze": "s.presenze",
            "media_voto": "s.media_voto",
            "fantamedia": "s.fantamedia",
            "gol": "s.gol",
            "gol_subiti": "s.gol_subiti",
            "rigori_parati": "s.rigori_parati",
            "rigori_calciati": "s.rigori_calciati",
            "bonus": "s.bonus",
            "malus": "s.malus",
            "assist": "s.assist",
            "ammonizioni": "s.ammonizioni",
            "espulsioni": "s.espulsioni",
            "autogol": "s.autogol",
        })
        .whenNotMatchedInsert(values={
            "giocatore_id": "s.giocatore_id",
            "stagione": "s.stagione",
            "presenze": "s.presenze",
            "media_voto": "s.media_voto",
            "fantamedia": "s.fantamedia",
            "gol": "s.gol",
            "gol_subiti": "s.gol_subiti",
            "rigori_parati": "s.rigori_parati",
            "rigori_calciati": "s.rigori_calciati",
            "bonus": "s.bonus",
            "malus": "s.malus",
            "assist": "s.assist",
            "ammonizioni": "s.ammonizioni",
            "espulsioni": "s.espulsioni",
            "autogol": "s.autogol",
        })
        .execute()
    )


# --------------------------------------------------------------------------
# squadre — UNIQUE(nome)
# --------------------------------------------------------------------------
def upsert_squadre(incoming: DataFrame) -> None:
    """Upsert squadre su nome."""
    spark = _get_spark()
    dt = DeltaTable.forName(spark, f"{CATALOG}.{SCHEMA}.squadre")
    (
        dt.alias("t")
        .merge(
            incoming.alias("s"),
            "t.nome = s.nome"
        )
        .whenMatchedUpdate(set={
            "allenatore": "s.allenatore",
            "budget_totale": "s.budget_totale",
        })
        .whenNotMatchedInsertAll()
        .execute()
    )


# --------------------------------------------------------------------------
# voti_giornata — UNIQUE(giocatore_id, giornata, stagione)
# --------------------------------------------------------------------------
def upsert_voti_giornata(incoming: DataFrame) -> None:
    """Upsert voti su (giocatore_id, giornata, stagione)."""
    spark = _get_spark()
    dt = DeltaTable.forName(spark, f"{CATALOG}.{SCHEMA}.voti_giornata")
    (
        dt.alias("t")
        .merge(
            incoming.alias("s"),
            "t.giocatore_id = s.giocatore_id "
            "AND t.giornata = s.giornata "
            "AND t.stagione = s.stagione"
        )
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )


# --------------------------------------------------------------------------
# asta_log — merge su giocatore_id
# --------------------------------------------------------------------------
def upsert_asta_log(incoming: DataFrame) -> None:
    """Upsert log asta su giocatore_id."""
    spark = _get_spark()
    dt = DeltaTable.forName(spark, f"{CATALOG}.{SCHEMA}.asta_log")
    (
        dt.alias("t")
        .merge(
            incoming.alias("s"),
            "t.giocatore_id = s.giocatore_id"
        )
        .whenMatchedUpdate(set={
            "prezzo_finale": "s.prezzo_finale",
            "squadra_acquirente": "s.squadra_acquirente",
            "fonte": "s.fonte",
        })
        .whenNotMatchedInsertAll()
        .execute()
    )


# --------------------------------------------------------------------------
# mia_rosa — merge su giocatore_id
# --------------------------------------------------------------------------
def upsert_mia_rosa(incoming: DataFrame) -> None:
    """Upsert rosa su giocatore_id."""
    spark = _get_spark()
    dt = DeltaTable.forName(spark, f"{CATALOG}.{SCHEMA}.mia_rosa")
    (
        dt.alias("t")
        .merge(
            incoming.alias("s"),
            "t.giocatore_id = s.giocatore_id"
        )
        .whenMatchedUpdate(set={
            "prezzo_pagato": "s.prezzo_pagato",
            "ruolo_fanta": "s.ruolo_fanta",
        })
        .whenNotMatchedInsertAll()
        .execute()
    )


# --------------------------------------------------------------------------
# calendario — UNIQUE logico su (giornata, squadra_casa, stagione)
# --------------------------------------------------------------------------
def upsert_calendario(incoming: DataFrame) -> None:
    """Upsert calendario su (giornata, squadra_casa, stagione)."""
    spark = _get_spark()
    dt = DeltaTable.forName(spark, f"{CATALOG}.{SCHEMA}.calendario")
    (
        dt.alias("t")
        .merge(
            incoming.alias("s"),
            "t.giornata = s.giornata "
            "AND t.squadra_casa = s.squadra_casa "
            "AND t.stagione = s.stagione"
        )
        .whenMatchedUpdate(set={
            "squadra_trasferta": "s.squadra_trasferta",
            "data_partita": "s.data_partita",
        })
        .whenNotMatchedInsertAll()
        .execute()
    )


# --------------------------------------------------------------------------
# formazioni_probabili — UNIQUE logico su (giornata, squadra)
# --------------------------------------------------------------------------
def upsert_formazioni_probabili(incoming: DataFrame) -> None:
    """Upsert formazioni su (giornata, squadra)."""
    spark = _get_spark()
    dt = DeltaTable.forName(spark, f"{CATALOG}.{SCHEMA}.formazioni_probabili")
    (
        dt.alias("t")
        .merge(
            incoming.alias("s"),
            "t.giornata = s.giornata AND t.squadra = s.squadra"
        )
        .whenMatchedUpdate(set={
            "modulo": "s.modulo",
            "titolari": "s.titolari",
            "fonte": "s.fonte",
            "creato_il": F.current_timestamp(),
        })
        .whenNotMatchedInsertAll()
        .execute()
    )
