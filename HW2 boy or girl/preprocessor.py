# our pre-processor module
# call the preprocess() function
import pandas as pd
def preprocess(df):
    ''' proceed data cleaning for the input df, return the cleaned df. '''
    df = dropIndexAndTimestamp(df)
    df.dropna(axis=0, how="all",inplace=True)
    df = nominalLabelEncode(df)
    df = selfIntroEncode(df)
    df = scaleOutliers(df)
    df = addBMI(df)
    df = binning(df)
    return df

# belows are the pre-processor methods
def dropIndexAndTimestamp(df):
    if {'index', 'Timestamp'}.issubset(df.columns):
        df.drop(['index', 'Timestamp'], axis=1, inplace=True)
    return df

def nominalLabelEncode(df):
    # lowercase string
    df['phone_os'] = df['phone_os'].str.lower()
    # removing leading and trailing whitespaces
    df['phone_os'] = df['phone_os'].str.strip()

    coded_star_signs = {'水瓶座':1, '雙魚座':2, '牡羊座':3, '金牛座':4, '雙子座':5, '巨蟹座':6, '獅子座':7, '處女座':8, '天秤座':9, '天蠍座':10, '射手座':11, '摩羯座':12}
    coded_phone_os = {'apple':1, 'android':2, 'windows phone':3, 'johncena':4}

    df = df.replace({"star_sign": coded_star_signs})
    df = df.replace({"phone_os": coded_phone_os})
    return df

def selfIntroEncode(df):
    df['new_self_intro'] = df['self_intro'].apply(lambda x: 1 if 'andsome' in str(x) else 0)
    df.drop(['self_intro'], axis=1, inplace=True)
    return df

def scaleOutliers(df):
    rule = [
            {'column':'height', 'max':200, 'min':145, 'maxValue':None, 'minValue':None,'to_num':True},
            {'column':'weight', 'max':100, 'min':45, 'maxValue':None, 'minValue':None,'to_num':True},
            {'column':'fb_friends', 'max':2000, 'min':0, 'maxValue':2000, 'minValue':0,'to_num':True},
            {'column':'yt', 'max':20, 'min':0, 'maxValue':None, 'minValue':0, 'to_num':True},
            {'column':'iq', 'max':200, 'min':0, 'maxValue':None, 'minValue':0, 'to_num':True}
    ]

    for r in rule:
        if r.get('to_num')!=None:
            df[r.get('column')] = pd.to_numeric(df[r.get('column')], errors='coerce')
        if r.get('max')!=None:
            indexRemove = df[df[r.get('column')]>r.get('max')].index
            df.loc[indexRemove, r.get('column')] = r.get('maxValue')
        if r.get('min')!=None:
            indexRemove = df[df[r.get('column')]<r.get('min')].index
            df.loc[indexRemove, r.get('column')] = r.get('minValue')
            
    for r in rule:
        hasValue = df[df[r.get('column')].notna()].index
        valueMean = df.loc[hasValue, r.get('column')].mean()
        df[r.get('column')] = df[r.get('column')].fillna(valueMean)
    return df

def addBMI(df):
    df['bmi'] = df['weight']/((df['height']/100) ** 2) # range from 11.25~95.125
    return df

def binning(df):
    # Bucketing values into bins
    # height	weight	iq	fb_friends	yt
    # range(min-1, max+range+1, range) 為了涵蓋頭尾，max自成一類。
    # min max值是copy上方rules的喔，記得兩邊一起改。
    bins = {
        'height': list(range(145-1, 200+5+1, 5)), #12 bins
        'weight': list(range(45-1, 100+5+1, 5)), #12 bins
        'iq':     list(range(0-1, 200+20+1, 20)), #10 bins
        'fb_friends': list(range(0-1, 2000+200+1, 200)), #10 bins
        'yt':     [-0.1, 0.5 ,1.0, 2.0, 4.0, 10.0, 20.0], # 6 bins
		'bmi':    list(range(12-1, 36+2+1, 2)), #13+1 bins
    }
    bins['bmi'].append(96) # 應付極端值 自成一類
    for key in bins.keys():
        df[key] = pd.cut(df[key], bins[key], labels=False)
    return df