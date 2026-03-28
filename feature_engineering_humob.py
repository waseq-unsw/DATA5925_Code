import pandas as pd
import numpy as np
import gensim
from gensim import corpora
from tqdm import tqdm

zones = ["Arts and Entertainment", "Business and Professional Services",
    "Community and Government", "Dining and Drinking", "Event",
    "Health and Medicine", "Landmarks and Outdoors", "Nightlife Spot",
    "Retail", "Sports and Recreation", "Travel and Transportation"]

weekend_list = [0, 1, 6, 7, 8, 13, 14, 20, 21, 27, 28, 29, 34, 35, 37, 41, 42, 
                48, 49, 50, 55, 56, 62, 63, 69, 70]

zone_mapping = {
    "Food": "Dining and Drinking",
    "Shopping": "Retail",
    "Entertainment": "Arts and Entertainment",
    "Japanese restaurant": "Dining and Drinking",
    "Western restaurant": "Dining and Drinking",
    "Eat all you can restaurant": "Dining and Drinking",
    "Chinese restaurant": "Dining and Drinking",
    "Indian restaurant": "Dining and Drinking",
    "Ramen restaurant": "Dining and Drinking",
    "Curry restaurant": "Dining and Drinking",
    "BBQ restaurant": "Dining and Drinking",
    "Hot pot restaurant": "Dining and Drinking",
    "Bar": "Dining and Drinking",
    "Diner": "Dining and Drinking",
    "Creative cuisine": "Dining and Drinking",
    "Organic cuisine": "Dining and Drinking",
    "Pizza": "Dining and Drinking",
    "Café": "Dining and Drinking",
    "Tea Salon": "Dining and Drinking",
    "Bakery": "Dining and Drinking",
    "Sweets ": "Dining and Drinking",
    "Wine Bar": "Dining and Drinking",
    "Pub": "Dining and Drinking",
    "Disco": "Nightlife Spot",
    "Beer Garden": "Dining and Drinking",
    "Fast Food": "Dining and Drinking",
    "Karaoke": "Arts and Entertainment", 
    "Cruising": "Travel and Transportation", 
    "Theme Park Restaurant": "Dining and Drinking",
    "Amusement Restaurant": "Dining and Drinking",
    "Other Restaurants": "Dining and Drinking",
    "Glasses": "Retail",
    "Drug Store": "Retail",
    "Electronics Store": "Retail",
    "DIY Store": "Retail",
    "Convenience Store": "Retail",
    "Recycle Shop": "Retail",
    "Interior Shop": "Retail",
    "Sports Store": "Retail",
    "Clothes Store": "Retail",
    "Grocery Store": "Retail",
    "Online Grocery Store": "Retail",
    "Sports Recreation": "Sports and Recreation",
    "Game Arcade": "Arts and Entertainment",
    "Swimming Pool": "Sports and Recreation",
    "Hotel": "Travel and Transportation",
    "Park": "Landmarks and Outdoors",
    "Transit Station": "Travel and Transportation",
    "Parking Area": "Travel and Transportation",
    "Casino": "Arts and Entertainment",
    "Hospital": "Health and Medicine",
    "Pharmacy": "Retail",
    "Chiropractic": "Health and Medicine",
    "Elderly Care Home": "Health and Medicine", 
    "Fishing": "Sports and Recreation",
    "School": "Community and Government",
    "Cram School": "Community and Government",
    "Kindergarten": "Community and Government",
    "Real Estate": "Business and Professional Services",
    "Home Appliances": "Retail",
    "Post Office": "Community and Government",
    "Laundry ": "Business and Professional Services",
    "Driving School": "Community and Government",
    "Wedding Ceremony": "Event", 
    "Cemetary": "Community and Government",
    "Bank": "Business and Professional Services",
    "Vet": "Health and Medicine",
    "Hot Spring": "Landmarks and Outdoors",
    "Hair Salon": "Business and Professional Services",
    "Lawyer Office": "Business and Professional Services",
    "Recruitment Office": "Business and Professional Services",
    "City Hall": "Community and Government",
    "Community Center": "Community and Government",
    "Church": "Community and Government",
    "Retail Store": "Retail",
    "Accountant Office": "Business and Professional Services",
    "IT Office": "Business and Professional Services",
    "Publisher Office": "Business and Professional Services",
    "Building Material": "Retail",
    "Gardening": "Business and Professional Services",
    "Heavy Industry": "Business and Professional Services",
    "NPO": "Community and Government",
    "Utility Copany": "Community and Government",
    "Port": "Travel and Transportation",
    "Research Facility": "Business and Professional Services",
}

def run_feature_engineering(mob_path, grid_path, poi_map_path, output_path):
    mob_df = pd.read_csv(mob_path)
    grid_df = pd.read_csv(grid_path)
    poi_map = pd.read_csv(poi_map_path, header=None)

    # --- PART 1: FUNCTIONAL ZONING (LDA) ---
    grid_df['POI_name'] = (grid_df['POIcategory'] - 1).map(poi_map.iloc[:,0])
    grid_df["zone"] = grid_df["POI_name"].map(zone_mapping).fillna("Other")

    grid_docs = []
    cell_keys = []
    for (x, y), g in grid_df.groupby(['x', 'y']):
        tokens = [row['zone'] for _, row in g.iterrows() for _ in range(int(row['POI_count']))]
        grid_docs.append(tokens)
        cell_keys.append((x, y))

    # Force vocabulary including "Event"
    dictionary = corpora.Dictionary(grid_docs)
    for z in zones:
        if z not in dictionary.token2id:
            dictionary.add_documents([[z]])

    corpus = [dictionary.doc2bow(doc) for doc in grid_docs]
    lda = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, passes=20, random_state=42)

    lda_vectors = []
    for doc in corpus:
        vec = np.zeros(5)
        for topic, prob in lda.get_document_topics(doc):
            vec[topic] = prob
        lda_vectors.append(vec)
    
    lda_df = pd.DataFrame(lda_vectors, columns=[f'lda_topic_{i}' for i in range(5)])
    lda_df[['x', 'y']] = cell_keys
    
    # Add Density
    poi_counts = grid_df.groupby(['x', 'y'])['POI_count'].sum().reset_index(name='poi_density')
    lda_df = lda_df.merge(poi_counts, on=['x', 'y'], how='left')

    # --- PART 2: Day Encoding ---
    mob_df['wd'] = mob_df['d'] % 7
    mob_df['is_weekend'] = mob_df['d'].isin(weekend_list).astype(int)

    # --- PART 3: PERSON TYPE ---
    # Identify Home (t < 12)
    home_locs = mob_df[mob_df['t'] < 12].groupby('uid')[['x', 'y']].agg(
        lambda x: x.mode()[0] if not x.mode().empty else -1
    ).reset_index().rename(columns={'x':'hx', 'y':'hy'})
    
    # Calculate Daily Radius/Complexity (N)
    # Using string join to avoid coordinate math errors
    mob_df['loc_id'] = mob_df['x'].astype(str) + "_" + mob_df['y'].astype(str)
    daily_n = mob_df.groupby(['uid', 'd'])['loc_id'].nunique().reset_index(name='daily_N')

    # --- PART 3: FINAL ASSEMBLY ---
    print("📦 Merging and Saving...")
    mob_df = mob_df.merge(lda_df, on=['x', 'y'], how='left').fillna(0)
    mob_df = mob_df.merge(home_locs, on='uid', how='left')
    mob_df = mob_df.merge(daily_n, on=['uid', 'd'], how='left')
    
    
    
    # Time Delta (Sequence Logic)
    mob_df = mob_df.sort_values(['uid', 'd', 't'])
    mob_df['time_delta'] = (mob_df['d'] * 48 + mob_df['t']).diff().fillna(0)
    # Cap delta at 47 as per your original logic
    mob_df.loc[mob_df['uid'] != mob_df['uid'].shift(), 'time_delta'] = 0
    mob_df['time_delta'] = mob_df['time_delta'].clip(upper=47)

    # Save to Parquet (Better than CSV for 100k records)
    mob_df.to_parquet(output_path, index=False)
    print(f"✅ Success! Enriched dataset saved to: {output_path}")

if __name__ == "__main__":
    run_feature_engineering(
        mob_path="yjmob100k-dataset1.csv.gz",
        grid_path="cell_POIcat.csv.gz",
        poi_map_path="POI_datacategories.csv",
        output_path="enriched_human_mobility_100k.parquet"
    )
