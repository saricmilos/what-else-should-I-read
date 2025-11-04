import pandas as pd

def split_location(df):
    location_split = df['Location'].str.split(',', expand=True)
    df['City'] = location_split[0].str.strip()
    df['State'] = location_split[1].str.strip()
    df['Country'] = location_split[2].str.strip()
    df[['City', 'State', 'Country']] = df[['City', 'State', 'Country']].fillna('Unknown')
    return df

def clean_country(df, country_mapping, region_mapping):
    df['Country'] = df['Country'].str.lower().str.strip()
    df['Country'] = df['Country'].map(lambda x: country_mapping.get(x, x))
    counts = df['Country'].value_counts()
    rare_countries = counts[counts <= 10].index
    df['Country_Clean'] = df['Country'].apply(lambda x: 'Other' if x in rare_countries else x)
    df['Country_Clean'] = df['Country_Clean'].str.strip().str.replace('"', '')
    df['Country_Clean'] = df['Country_Clean'].replace('', 'unknown')
    df['Region'] = df['Country_Clean'].map(lambda x: region_mapping.get(x, 'Other'))
    df = df.drop(columns=['Country'])
    return df

def clean_city(df, top_n=50):
    df['City_Clean'] = df['City'].str.strip().str.lower()
    df['City_Clean'] = df['City_Clean'].replace({'': 'unknown', 'n/a': 'unknown'})
    city_counts = df['City_Clean'].value_counts()
    top_cities = city_counts.nlargest(top_n).index
    df['City_Clean'] = df['City_Clean'].apply(lambda x: x if x in top_cities else 'Other')
    df = df.drop(columns=['City'])
    return df

def clean_state(df, top_n=50):
    df['State_Clean'] = df['State'].str.strip().str.lower()
    df['State_Clean'] = df['State_Clean'].replace(['', 'n/a', '\\n/a"', 'na'], 'Other')
    top_states = df['State_Clean'].value_counts().nlargest(top_n).index
    df['State_Clean'] = df['State_Clean'].apply(lambda x: x if x in top_states else 'Other')
    df = df.drop(columns=['State'])
    return df

def preprocess_location(df, country_mapping, region_mapping, top_cities=50, top_states=50):
    df = split_location(df)
    df = clean_country(df, country_mapping, region_mapping)
    df = clean_city(df, top_cities)
    df = clean_state(df, top_states)
    return df
