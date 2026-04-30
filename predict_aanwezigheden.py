"""
Gebruik:
    python predict_aanwezigheden.py invoer.csv

Het invoer-CSV-bestand moet deze kolommen bevatten:
    datum           YYYY-MM-DD
    starttijd       HH:MM
    eindtijd        HH:MM
    klascode        bijv. MLG301
    programma       naam van het programma
    lokaalcode      bijv. B1.015
    activiteit      CanonicalActivity-waarde uit de database
    verwacht_aantal aantal verwachte studenten

Uitvoer: resultaten worden getoond in de terminal en opgeslagen als 'voorspellingen.csv'.
"""

import sys
import argparse
from datetime import date, timedelta, datetime

import numpy as np
import pandas as pd
import requests
import joblib


MODEL_PAD = 'model_aanwezigheden.pkl'


# ---------------------------------------------------------------------------
# Hulpfuncties (zelfde logica als in het notebook)
# ---------------------------------------------------------------------------

def dagdeel(hour):
    if hour < 12:
        return 'ochtend'
    elif hour < 17:
        return 'namiddag'
    return 'avond'


def haal_weer_op(datums: list[date]) -> pd.DataFrame:
    """Haalt dagelijkse weerdata op voor Gent via Open-Meteo."""
    start = min(datums).strftime('%Y-%m-%d')
    einde = max(datums).strftime('%Y-%m-%d')

    # Gebruik historisch archief voor data in het verleden, forecast voor de toekomst
    vandaag = date.today()
    if max(datums) <= vandaag:
        url = 'https://archive-api.open-meteo.com/v1/archive'
    else:
        url = 'https://api.open-meteo.com/v1/forecast'

    params = {
        'latitude':   51.05,
        'longitude':   3.72,
        'start_date': start,
        'end_date':   einde,
        'daily':      'temperature_2m_mean,precipitation_sum,weathercode',
        'timezone':   'Europe/Brussels',
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        weer_df = pd.DataFrame({
            'datum':    pd.to_datetime(data['daily']['time']).dt.date,
            'gem_temp': data['daily']['temperature_2m_mean'],
            'neerslag': data['daily']['precipitation_sum'],
            'weercode': data['daily']['weathercode'],
        })
        weer_df['is_regen'] = (weer_df['weercode'] >= 51).astype(int)
        return weer_df
    except Exception as e:
        print(f'[WAARSCHUWING] Weerdata ophalen mislukt: {e}')
        print('[WAARSCHUWING] Gemiddelde waarden worden gebruikt.')
        rows = [{'datum': d, 'gem_temp': 10.0, 'neerslag': 1.5, 'weercode': 3, 'is_regen': 0}
                for d in datums]
        return pd.DataFrame(rows)


def bereken_vakantie_features(d: date, alle_vakantiedagen: set) -> tuple[int, int]:
    """Geeft (dagen_tot_vakantie, dagen_na_vakantie) terug voor datum d."""
    dagen_tot = 14
    for i in range(1, 15):
        if d + timedelta(days=i) in alle_vakantiedagen:
            dagen_tot = i
            break

    dagen_na = 14
    for i in range(1, 15):
        if d - timedelta(days=i) in alle_vakantiedagen:
            dagen_na = i
            break

    return dagen_tot, dagen_na


# ---------------------------------------------------------------------------
# Hoofdfunctie
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Voorspel aanwezigheid voor een lijst lessen.')
    parser.add_argument('invoer_csv', help='Pad naar het invoer-CSV-bestand')
    args = parser.parse_args()

    # --- Model laden ---
    print(f'Model laden uit {MODEL_PAD} ...')
    artifact = joblib.load(MODEL_PAD)
    model             = artifact['model']
    room_lookup       = artifact['room_lookup']
    class_lookup      = artifact['class_lookup']
    feature_cols_num  = artifact['feature_cols_num']
    feature_cols_cat  = artifact['feature_cols_cat']
    alle_vakantiedagen = artifact['alle_vakantiedagen']
    semester_start    = artifact['semester_start']

    # --- Invoer lezen ---
    print(f'Invoer lezen uit {args.invoer_csv} ...')
    df = pd.read_csv(args.invoer_csv)

    vereiste_kolommen = ['datum', 'starttijd', 'eindtijd', 'klascode',
                         'programma', 'lokaalcode', 'activiteit', 'verwacht_aantal']
    ontbrekend = [k for k in vereiste_kolommen if k not in df.columns]
    if ontbrekend:
        print(f'[FOUT] Ontbrekende kolommen in CSV: {ontbrekend}')
        sys.exit(1)

    df['datum']      = pd.to_datetime(df['datum']).dt.date
    df['starttijd']  = pd.to_datetime(df['starttijd'],  format='%H:%M').dt.time
    df['eindtijd']   = pd.to_datetime(df['eindtijd'],   format='%H:%M').dt.time

    # --- Weerdata ophalen ---
    print('Weerdata ophalen ...')
    weer_df = haal_weer_op(df['datum'].tolist())
    df = df.merge(weer_df, on='datum', how='left')

    # --- Vakantie-features ---
    vakantie = df['datum'].apply(lambda d: bereken_vakantie_features(d, alle_vakantiedagen))
    df['dagen_tot_vakantie'] = vakantie.apply(lambda x: x[0])
    df['dagen_na_vakantie']  = vakantie.apply(lambda x: x[1])

    # --- Tijdsfeatures ---
    df['FromHour']    = df['starttijd'].apply(lambda t: t.hour)
    df['FromMinutes'] = df['starttijd'].apply(lambda t: t.minute)
    df['UntilHour']   = df['eindtijd'].apply(lambda t: t.hour)
    df['UntilMinutes']= df['eindtijd'].apply(lambda t: t.minute)
    df['les_duur_min']= (df['UntilHour'] * 60 + df['UntilMinutes']) - (df['FromHour'] * 60 + df['FromMinutes'])
    df['dagdeel']     = df['FromHour'].apply(dagdeel)

    df_datum = pd.to_datetime(df['datum'].astype(str))
    df['Weekday']          = df_datum.dt.weekday + 1
    df['Month']            = df_datum.dt.month
    df['week_in_semester'] = ((df_datum - semester_start).dt.days // 7 + 1).clip(1, 20)

    # --- Lookups: klas en lokaal ---
    df['ClassCode']  = df['klascode']
    df['ProgramName']= df['programma']
    df['CanonicalActivity'] = df['activiteit']

    df['ClassCredits'] = df['klascode'].map(
        lambda k: class_lookup.get(k, {}).get('ClassCredits', np.nan)
    )

    df['RoomCategory'] = df['lokaalcode'].map(
        lambda r: room_lookup.get(r, {}).get('RoomCategory', 'Onbekend')
    )
    df['Capacity'] = df['lokaalcode'].map(
        lambda r: room_lookup.get(r, {}).get('Capacity', np.nan)
    )

    # Bekende activiteitstypes → IsCourse / IsExam / IsPractical
    # Standaard 0 als onbekend; het model is getraind op deze waarden
    df['IsCourse']    = df['activiteit'].str.lower().str.contains('course|les|hoorcollege', na=False).astype(int)
    df['IsExam']      = df['activiteit'].str.lower().str.contains('exam|examen', na=False).astype(int)
    df['IsPractical'] = df['activiteit'].str.lower().str.contains('practical|practicum|oefening', na=False).astype(int)

    # Verwachte bezettingsgraad
    df['verwachte_bezetting'] = (
        df['verwacht_aantal'] / df['Capacity'].replace(0, np.nan)
    ).clip(0, 2)

    # --- Features samenvoegen ---
    X = df[feature_cols_num + feature_cols_cat].copy()

    # --- Voorspellen ---
    print('Voorspellingen berekenen ...')
    df['voorspelde_aanwezigheidsgraad'] = model.predict(X).clip(0, 1.5)
    df['voorspeld_aantal'] = (
        df['voorspelde_aanwezigheidsgraad'] * df['verwacht_aantal']
    ).round().astype(int)

    # --- Nauwkeurigheidsschatting ---
    # Als werkelijke aantallen beschikbaar zijn, bereken dan MAE
    if 'werkelijk_aantal' in df.columns:
        from sklearn.metrics import mean_absolute_error
        mae = mean_absolute_error(df['werkelijk_aantal'], df['voorspeld_aantal'])
        print(f'\nMAE op opgegeven werkelijke aantallen: {mae:.1f} studenten')

    # --- Resultaten tonen ---
    kolommen_uitvoer = [
        'datum', 'starttijd', 'klascode', 'programma', 'lokaalcode',
        'verwacht_aantal', 'voorspelde_aanwezigheidsgraad', 'voorspeld_aantal',
    ]
    resultaten = df[kolommen_uitvoer].copy()
    resultaten['voorspelde_aanwezigheidsgraad'] = resultaten['voorspelde_aanwezigheidsgraad'].round(3)

    print('\n--- Resultaten ---')
    print(resultaten.to_string(index=False))

    uitvoer_pad = 'voorspellingen.csv'
    resultaten.to_csv(uitvoer_pad, index=False)
    print(f'\nResultaten opgeslagen in {uitvoer_pad}')


if __name__ == '__main__':
    main()
