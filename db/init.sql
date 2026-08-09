-- FantAssistant - schema iniziale

-- Anagrafica giocatori
CREATE TABLE IF NOT EXISTS giocatori (
    id SERIAL PRIMARY KEY,
    fanta_id INTEGER UNIQUE, -- Id ufficiale usato nei listoni Fantacalcio, utile per aggiornamenti futuri
    nome TEXT NOT NULL,
    ruolo TEXT NOT NULL CHECK (ruolo IN ('P', 'D', 'C', 'A')),
    squadra TEXT NOT NULL,
    quotazione_iniziale NUMERIC,
    quotazione_attuale NUMERIC,
    fvm NUMERIC, -- Fantavalore di Mercato, utile per orientarsi in asta
    creato_il TIMESTAMP DEFAULT now(),
    UNIQUE (nome, squadra)
);

-- Statistiche storiche per stagione
CREATE TABLE IF NOT EXISTS statistiche_storiche (
    id SERIAL PRIMARY KEY,
    giocatore_id INTEGER REFERENCES giocatori(id) ON DELETE CASCADE,
    stagione TEXT NOT NULL,
    presenze INTEGER DEFAULT 0,
    gol INTEGER DEFAULT 0,
    assist INTEGER DEFAULT 0,
    media_voto NUMERIC,
    ammonizioni INTEGER DEFAULT 0,
    espulsioni INTEGER DEFAULT 0,
    UNIQUE (giocatore_id, stagione)
);

-- Calendario partite
CREATE TABLE IF NOT EXISTS calendario (
    id SERIAL PRIMARY KEY,
    giornata INTEGER NOT NULL,
    squadra_casa TEXT NOT NULL,
    squadra_trasferta TEXT NOT NULL,
    data_partita TIMESTAMP,
    stagione TEXT NOT NULL
);

-- Formazioni probabili/ufficiali
CREATE TABLE IF NOT EXISTS formazioni_probabili (
    id SERIAL PRIMARY KEY,
    giornata INTEGER NOT NULL,
    squadra TEXT NOT NULL,
    modulo TEXT,
    titolari TEXT[], -- lista nomi giocatori
    fonte TEXT, -- 'vision' | 'manuale' | 'etl'
    creato_il TIMESTAMP DEFAULT now()
);

-- Voti giornata (post-partita)
CREATE TABLE IF NOT EXISTS voti_giornata (
    id SERIAL PRIMARY KEY,
    giocatore_id INTEGER REFERENCES giocatori(id) ON DELETE CASCADE,
    giornata INTEGER NOT NULL,
    stagione TEXT NOT NULL,
    voto NUMERIC,
    fantavoto NUMERIC,
    gol INTEGER DEFAULT 0,
    assist INTEGER DEFAULT 0,
    ammonizione BOOLEAN DEFAULT FALSE,
    espulsione BOOLEAN DEFAULT FALSE,
    UNIQUE (giocatore_id, giornata, stagione)
);

-- La tua rosa
CREATE TABLE IF NOT EXISTS mia_rosa (
    id SERIAL PRIMARY KEY,
    giocatore_id INTEGER REFERENCES giocatori(id) ON DELETE CASCADE,
    prezzo_pagato NUMERIC,
    ruolo_fanta TEXT,
    acquistato_il TIMESTAMP DEFAULT now()
);

-- Log dell'asta (chi ha preso cosa, anche avversari - utile per sapere chi e' libero)
CREATE TABLE IF NOT EXISTS asta_log (
    id SERIAL PRIMARY KEY,
    giocatore_id INTEGER REFERENCES giocatori(id) ON DELETE CASCADE,
    prezzo_finale NUMERIC,
    squadra_acquirente TEXT,
    fonte TEXT, -- 'vision' | 'manuale'
    creato_il TIMESTAMP DEFAULT now()
);

-- Censimento squadre della lega, con budget totale (uguale per tutte all'inizio)
CREATE TABLE IF NOT EXISTS squadre (
    id SERIAL PRIMARY KEY,
    nome TEXT NOT NULL UNIQUE,
    budget_totale NUMERIC NOT NULL,
    creato_il TIMESTAMP DEFAULT now()
);