import pandas as pd
import numpy as np
import gensim
from gensim import corpora
from tqdm import tqdm

zones = ["Arts and Entertainment", "Business and Professional Services",
    "Community and Government", "Dining and Drinking", "Event",
    "Health and Medicine", "Landmarks and Outdoors", "Nightlife Spot",
    "Retail", "Sports and Recreation", "Travel and Transportation"]

holiday_list = [0, 1, 6, 7, 8, 13, 14, 20, 21, 27, 28, 29, 34, 35, 37, 41, 42, 
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
    "Disco": "Arts and Entertainment",
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
    "Pharmacy": "Health and Medicine",
    "Chiropractic": "Health and Medicine",
    "Elderly Care Home": "Community and Government", 
    "Fishing": "Sports and Recreation",
    "School": "Community and Government",
    "Cram School": "Community and Government",
    "Kindergarten": "Community and Government",
    "Real Estate": "Business and Professional Services",
    "Home Appliances": "Retail",
    "Post Office": "Community and Government",
    "Laundry ": "Business and Professional Services",
    "Driving School": "Community and Government",
    "Wedding Ceremony": "Business and Professional Services", 
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
    "Gardening": "Retail",
    "Heavy Industry": "Business and Professional Services",
    "NPO": "Community and Government",
    "Utility Copany": "Business and Professional Services",
    "Port": "Travel and Transportation",
    "Research Facility": "Business and Professional Services"
}

motif_map = {
    'Stayed Home': 0,
    'Rule I': 1,
    'Rule II': 2,
    'Rule III': 3,
    'Rule IV': 4,
    'Complex/Other': 5,
    'Unknown': 6
}

def classify_motif(row):
    N = row['N']
    stops = sorted(row['stops']) # Sort lengths to easily match the rules mathematically
    
    if N <= 1:
        return 'Stayed Home'
        
    num_tours = len(stops)
    
    # Rule II: 1 tour with N-1 stops
    if num_tours == 1 and stops == [N - 1]:
        return 'Rule II'
        
    elif num_tours == 2:
        # Rule I: 1 tour with 1 stop, 1 tour with N-2 stops
        if stops == sorted([1, N - 2]):
            return 'Rule I'
        # Rule IV: 1 tour with 2 stops, 1 tour with N-3 stops
        elif stops == sorted([2, N - 3]):
            return 'Rule IV'
            
    # Rule III: 2 tours with 1 stop, 1 tour with N-3 stops
    elif num_tours == 3:
        if stops == sorted([1, 1, N - 3]):
            return 'Rule III'
            
    return 'Complex/Other' # For rare motifs outside the primary 17


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

    n_topics = 5
    lda = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=n_topics, passes=20, random_state=42)

    lda_vectors = []
    for doc in corpus:
        vec = np.zeros(n_topics)
        for topic, prob in lda.get_document_topics(doc):
            vec[topic] = prob
        lda_vectors.append(vec)
    
    lda_df = pd.DataFrame(lda_vectors, columns=[f'lda_topic_{i}' for i in range(n_topics)])
    lda_df[['x', 'y']] = cell_keys
    
    # Add Density
    poi_counts = grid_df.groupby(['x', 'y'])['POI_count'].sum().reset_index(name='poi_density')
    lda_df = lda_df.merge(poi_counts, on=['x', 'y'], how='left')

    # --- PART 2: Day Encoding ---
    mob_df['wd'] = mob_df['d'] % 7
    mob_df['is_weekend'] = mob_df['wd'].isin([0, 6]).astype(int)
    mob_df['is_holiday'] = mob_df['d'].isin(holiday_list).astype(int)
    
    # --- PART 3: PERSON TYPE ---
    # Pre-calculate loc_id for the whole dataframe
    mob_df['loc_id'] = mob_df['x'].astype(str) + "_" + mob_df['y'].astype(str)

    # 1. Identify Home
    night_df = mob_df[mob_df['t'] < 12]
    # Count frequency of each location per user
    home_counts = night_df.groupby(['uid', 'loc_id','x','y']).size().reset_index(name='visits')
    anchor_homes = home_counts.sort_values(['uid', 'visits'], ascending=[True, False]).drop_duplicates('uid', keep='first')
    anchor_homes = anchor_homes.rename(columns={
        'x': 'anchor_x', 
        'y': 'anchor_y', 
        'loc_id': 'anchor_id'
    })
    #Merge Anchor back to all night locations to calculate spatial distance
    home_candidates = pd.merge(home_counts, anchor_homes[['uid', 'anchor_x', 'anchor_y', 'anchor_id']], on='uid')

    #Define the "Home Cluster" (Any night location within 1 grid cell of the Anchor)
    home_candidates['is_home_cell'] = (
        (abs(home_candidates['x'] - home_candidates['anchor_x']) <= 1) & 
        (abs(home_candidates['y'] - home_candidates['anchor_y']) <= 1)
    )

    #Extract a mapping of which messy locations belong to the clean Anchor
    home_cluster_map = home_candidates[home_candidates['is_home_cell']][['uid', 'loc_id', 'anchor_id', 'anchor_x', 'anchor_y', 'is_home_cell']]
    mob_df = pd.merge(mob_df, home_cluster_map, on=['uid', 'loc_id'], how='left')
    
    #Assign States: If it's in the Home Cluster, it's 'H'. Otherwise, 'T'.
    mob_df['state'] = np.where(mob_df['is_home_cell'].notna(), 'H', 'T')
    mob_df['loc_id'] = np.where(mob_df['state'] == 'H', mob_df['anchor_id'], mob_df['loc_id'])
    mob_df['x'] = np.where(mob_df['state'] == 'H', mob_df['anchor_x'], mob_df['x']).astype(int)
    mob_df['y'] = np.where(mob_df['state'] == 'H', mob_df['anchor_y'], mob_df['y']).astype(int)

    # Clean up temporary columns
    mob_df = mob_df.drop(columns=['anchor_id', 'anchor_x', 'anchor_y', 'is_home_cell'])

    mob_df = mob_df.sort_values(by=['uid', 'd', 't'])
    mob_df['prev_loc'] = mob_df.groupby(['uid', 'd'])['loc_id'].shift()
    df_seq = mob_df[mob_df['loc_id'] != mob_df['prev_loc']].copy()
    
    # 4. NEW: Calculate N based on sequence stops
    # N = 1 (Home) + Total sequential 'T' stops made that day
    df_seq['is_transit'] = (df_seq['state'] == 'T')
    t_counts = df_seq.groupby(['uid', 'd'])['is_transit'].sum().reset_index(name='T_count')
    t_counts['N'] = t_counts['T_count'] + 1
    daily_n = t_counts[['uid', 'd', 'N']]
    df_seq = df_seq.drop(columns=['is_transit'])

    # 5. Extract Tours (Groups of 'T' separated by 'H')
    df_seq['is_home'] = (df_seq['state'] == 'H')
    df_seq['tour_id'] = df_seq.groupby(['uid', 'd'])['is_home'].cumsum()
    # Count stops in each tour
    tours_only = df_seq[df_seq['state'] == 'T']
    tour_lengths = tours_only.groupby(['uid', 'd', 'tour_id']).size().reset_index(name='stops')
    # Group lengths into a list for classification
    daily_tours = tour_lengths.groupby(['uid', 'd'])['stops'].apply(list).reset_index()

    # 6. Merge and Classify
    daily_motifs = pd.merge(daily_n, daily_tours, on=['uid', 'd'], how='left')
    daily_motifs['stops'] = daily_motifs['stops'].apply(lambda x: x if isinstance(x, list) else [])
    daily_motifs['daily_rule'] = daily_motifs.apply(classify_motif, axis=1)
    daily_motifs['motif_id'] = 'motif_' + daily_motifs['daily_rule'].map(motif_map).astype(str)

    # 7. Extract dominant characteristic motif per user
    training_days = daily_motifs[daily_motifs['d'] < 60]
    motif_counts = training_days.groupby(['uid', 'motif_id']).size().unstack(fill_value=0)
    motif_distribution = motif_counts.div(motif_counts.sum(axis=1), axis=0)
    
    #user_category = training_days.groupby('uid')['daily_rule'].apply(
    #    lambda x: x.value_counts().idxmax()
    #).reset_index(name='characteristic_motif')

    # --- PART 3: FINAL ASSEMBLY ---
    # Merge LDA features
    mob_df = mob_df.merge(lda_df, on=['x', 'y'], how='left').fillna(0)
    # Merge the Person Type
    expected_rules = ['motif_0', 'motif_1', 'motif_2', 'motif_3', 'motif_4', 'motif_5', 'motif_6']
    mob_df = mob_df.merge(motif_distribution, on='uid', how='left')
    missing_cols = list(set(expected_rules) - set(mob_df.columns))
    mob_df[missing_cols] = 0.0 
    # 2. Handle missing users (fill NaNs with 0.0 for the neural network)
    # This replaces your old .fillna('Unknown') logic
    mob_df[expected_rules] = mob_df[expected_rules].fillna(0.0)

    # Time Delta (Sequence Logic)
    mob_df = mob_df.sort_values(['uid', 'd', 't'])
    mob_df['time_delta'] = (mob_df['d'] * 48 + mob_df['t']).diff().fillna(0)
    # Cap delta at 47 as per your original logic
    mob_df.loc[mob_df['uid'] != mob_df['uid'].shift(), 'time_delta'] = 0
    mob_df['time_delta'] = mob_df['time_delta'].clip(upper=47)

    # Drop extra columns
    cols_to_drop = ['loc_id', 'home_id', 'state', 'prev_loc']
    mob_df = mob_df.drop(columns=cols_to_drop, errors='ignore')

    # Save to Parquet (Better than CSV for 100k records)
    mob_df.to_parquet(output_path, index=False)

if __name__ == "__main__":
    run_feature_engineering(
        mob_path="yjmob100k-dataset1.csv.gz",
        grid_path="cell_POIcat.csv.gz",
        poi_map_path="POI_datacategories.csv",
        output_path="enriched_human_mobility_100k.parquet"
    )
