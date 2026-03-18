

print(f'Processing : Dataset_load...')
print('')
 
# Load dataset
df = dataset_load(Config.task, Config.debug, INPUT_DIR)
 
all_days = np.arange(df['d'].nunique())
weekend_list = [0, 1, 6, 7, 8, 13, 14, 20, 21, 27, 28, 29, 34, 35, 37, 41, 42, 48, 49, 50, 55, 56, 62, 63, 69, 70,]
weekday_list = [day for day in all_days if day not in weekend_list]
 
# preprocess
df = extract_uid(df, Config.start_uid, Config.end_uid)
df["wd"] = df["d"] % 7
df['xy'] = df['x'].astype(str).str.zfill(3) + df['y'].astype(str).str.zfill(3)
df['xy'] = df['xy'].astype(int)
 
df = add_cumcount(df)
df = add_group_xx(df, window_size=Config.window_size)
df, drop_cols = add_lag_mesh(df, Config.n_ago)
 
train, test = train_test_split_func(df, Config.train_start_day, Config.train_end_day, Config.test_start_day, Config.test_end_day)