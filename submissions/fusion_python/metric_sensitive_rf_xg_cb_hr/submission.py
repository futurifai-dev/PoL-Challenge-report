# -*- coding: utf-8 -*-
"""heuristic_baseline.ipynb

Automatically generated.

Original file is located at:
    /home/victor/Github/splid-devkit/baseline_submissions/heuristic_python/heuristic_baseline.ipynb
"""

# import time
from pathlib import Path
import pickle
import pandas as pd

import utils

import pandas as pd
import numpy as np
import os
from node import Node
from datetime import datetime, timedelta
from pathlib import Path
from catboost import CatBoostClassifier, Pool
import sys
import time

# INPUT/OUTPUT PATHS WITHIN THE DOCKER CONTAINER
TRAINED_MODEL_DIR = Path('/trained_model/')
TEST_DATA_DIR = Path('/dataset/test/')
TEST_PREDS_FP = Path('/submission/submission.csv')

# INPUT/OUTPUT PATHS WITHIN THE DOCKER CONTAINER
# TEST_DATA_DIR = Path('../../../data/phase_1_v3/fold_0/valid')
# TEST_PREDS_FP = Path('../../../data/submission/submission_fuse_metric_sensitive_local.csv')

class index_dict:
    def __init__(self):
        self.times = self.IDADIK()
        self.indices = []
        self.AD_dex =[]
        self.modes = self.mode()

    class IDADIK:
        def __init__(self):
            self.ID = []
            self.AD = []
            self.IK = []

    class mode:
        def __init__(self):
            self.SK = []
            self.end = []

datalist = []

# Searching for training data within the dataset folder
for file in os.listdir(TEST_DATA_DIR):
    if file.endswith(".csv"):
        datalist.append(os.path.join(TEST_DATA_DIR, file))

# Sort the training data and labels
datalist = sorted(datalist)

# Print the sorted filepath to the training data
print(datalist)

frames = list()

for idx_data in range(len(datalist)):
    detected = index_dict()
    filtered = index_dict()
    
    data_path = datalist[idx_data]
    data = pd.read_csv(data_path)
    
    # Read the objectID from the filename
    filename = data_path.split('/')[-1]
    
    satcat=int(filename.split('.')[0])
    
    # Extracting longitudinal and inclination information from the pandas dataframe
    longitudes = data["Longitude (deg)"]
    inclinations = data["Inclination (deg)"]
    
    # Arbitrary assign start time and end time. Note: SNICT was developed to read in time-stamped data, 
    # however, our training data are not label with a time stamp, hence an arbitrary start and end time
    # are selected
    starttime = datetime.fromisoformat("2023-01-01 00:00:00+00:00")
    endtime = datetime.fromisoformat("2023-07-01 00:00:00+00:00")
    
    # Get std for longitude over a 24 hours window
    lon_std = []
    nodes = []
    steps_per_day = 12
    lon_was_baseline = True
    lon_baseline = 0.03
    
    for i in range(len(data["Longitude (deg)"])):
        if i <= steps_per_day:
            lon_std.append(np.std(data["Longitude (deg)"][0:steps_per_day]))
        else:
            lon_std.append(np.std(data["Longitude (deg)"][i-steps_per_day:i]))
    
    ssEW = Node(satcat=satcat,
                t=starttime,
                index=0,
                ntype="SS",
                signal="EW")
    es = Node(satcat=satcat,
                t=endtime,
                index=len(data["Longitude (deg)"])-1,
                ntype="ES",
                signal="ES",
                mtype="ES")
    
    # Run LS detection
    for i in range(steps_per_day+1,len(lon_std)-steps_per_day):             # if at least 1 day has elapsed since t0
        max_lon_std_24h = np.max(lon_std[i-steps_per_day:i])
        min_lon_std_24h = np.min(lon_std[i-steps_per_day:i])
        A = np.abs(max_lon_std_24h-min_lon_std_24h)/2
        th_ = 0.95*A
    
        # ID detection
        if (lon_std[i]>lon_baseline) & lon_was_baseline:                    # if sd is elevated & last sd was at baseline
            before = np.mean(data["Longitude (deg)"][i-steps_per_day:i])    # mean of previous day's longitudes
            after = np.mean(data["Longitude (deg)"][i:i+steps_per_day])     # mean of next day's longitudes
            # if not temporary noise, then real ID
            if np.abs(before-after)>0.3:                                    # if means are different
                lon_was_baseline = False                                    # the sd is not yet back at baseline
                index = i
                if i < steps_per_day+2:
                    ssEW.mtype = "NK"
                else:
                    detected.times.ID.append(starttime+timedelta(hours=i*2))
        # IK detection
        elif (lon_std[i]<=lon_baseline) & (not lon_was_baseline):           # elif sd is not elevated and drift has already been initialized
            drift_ended = True                                              # toggle end-of-drift boolean 
            for j in range(steps_per_day):                                  # for the next day, check...
                if np.abs(data["Longitude (deg)"][i]-data["Longitude (deg)"][i+j])>0.3:       # if the longitude changes from the current value
                    drift_ended = False                                     # the drift has not ended
            if drift_ended:                                                 # if the drift has ended
                lon_was_baseline = True                                     # the sd is back to baseline
                detected.times.IK.append(starttime+timedelta(hours=i*2))    # save tnow as end-of-drift
                detected.indices.append([index,i])                          # save indices for t-start & t-end
    
        # Last step
        elif (i == (len(lon_std)-steps_per_day-1))\
            &(not lon_was_baseline):
            detected.times.IK.append(starttime+timedelta(hours=i*2))
            detected.indices.append([index,i])
    
        # AD detection
        elif ((lon_std[i]-max_lon_std_24h>th_) or (min_lon_std_24h-lon_std[i]>th_)) & (not lon_was_baseline):          # elif sd is elevated and drift has already been initialized
            if i >= steps_per_day+3:
                detected.times.AD.append(starttime+timedelta(hours=i*2))
                detected.AD_dex.append(i)
    
    def add_node(n):
        nodes[len(nodes)-1].char_mode(
            next_index = n.index,
            lons = longitudes,
            incs = inclinations
        )
        if n.type == "AD":
            nodes[len(nodes)-1].mtype = "NK"
    
        if (nodes[len(nodes)-1].mtype != "NK"):
            filtered.indices.append([nodes[len(nodes)-1].index,n.index])
            filtered.modes.SK.append(nodes[len(nodes)-1].mtype)
            stop_NS = True if n.type == "ID" else False
            filtered.modes.end.append(stop_NS)
        nodes.append(n)
    
    toggle = True
    nodes.append(ssEW)
    if len(detected.times.IK) == 1:
        if len(detected.times.ID) == 1:
            filtered.times.ID.append(detected.times.ID[0])                                  # keep the current ID
            ID = Node(satcat,
                    detected.times.ID[0],
                    index=detected.indices[0][0],
                    ntype='ID',
                    lon=longitudes[detected.indices[0][0]],
                    signal="EW")
            add_node(ID)
        filtered.times.IK.append(detected.times.IK[0]) 
        IK = Node(satcat,
                detected.times.IK[0],
                index=detected.indices[0][1],
                ntype='IK',
                lon=longitudes[detected.indices[0][1]],
                signal="EW")
        apnd = True
        if len(detected.times.AD) == 1:
            AD = Node(satcat,
                      detected.times.AD[0],
                      index=detected.AD_dex[0],
                      ntype="AD",
                      signal="EW")
            add_node(AD)
        elif len(detected.times.AD) == 0:
            pass
        else:
            for j in range(len(detected.times.AD)):
                ad = Node(satcat,
                      detected.times.AD[j],
                      index=detected.AD_dex[j],
                      ntype="AD",
                      signal="EW")
                ad_next = Node(satcat,
                      detected.times.AD[j+1],
                      index=detected.AD_dex[j+1],
                      ntype="AD",
                      signal="EW") \
                    if j < (len(detected.times.AD)-1) else None
                if (ad.t>starttime+timedelta(hours=detected.indices[0][0]*2))&(ad.t<IK.t):
                    if apnd & (ad_next is not None):
                        if ((ad_next.t-ad.t)>timedelta(hours=24)):
                            add_node(ad)
                        else:
                            add_node(ad)
                            apnd = False
                    elif apnd & (ad_next is None):
                        add_node(ad)
                    elif (not apnd) & (ad_next is not None):
                        if ((ad_next.t-ad.t)>timedelta(hours=24)):
                            apnd = True
        if detected.indices[0][1] != (len(lon_std)-steps_per_day-1):
            add_node(IK)    
    
    for i in range(len(detected.times.IK)-1):                                 # for each longitudinal shift detection
        if toggle:                                                            # if the last ID was not discarded
            if ((starttime+timedelta(hours=detected.indices[i+1][0]*2)-detected.times.IK[i])>timedelta(hours=36)):# if the time between the current IK & next ID is longer than 48 hours
                filtered.times.ID.append(detected.times.ID[i])                # keep the current ID
                filtered.times.IK.append(detected.times.IK[i])                # keep the current IK
                ID = Node(satcat,
                        detected.times.ID[i],
                        index=detected.indices[i][0],
                        ntype='ID',
                        lon=longitudes[detected.indices[i][0]],
                        signal="EW")
                add_node(ID)
                IK = Node(satcat,
                        detected.times.IK[i],
                        index=detected.indices[i][1],
                        ntype='IK',
                        lon=longitudes[detected.indices[i][1]],
                        signal="EW")
                apnd = True
                for j in range(len(detected.times.AD)):
                    ad = Node(satcat,
                      detected.times.AD[j],
                      index=detected.AD_dex[j],
                      ntype="AD",
                      signal="EW")
                    ad_next = Node(satcat,
                      detected.times.AD[j+1],
                      index=detected.AD_dex[j+1],
                      ntype="AD",
                      signal="EW") \
                        if j < (len(detected.times.AD)-1) else None
                    if (ad.t>ID.t)&(ad.t<IK.t):
                        if apnd & (ad_next is not None):
                            if ((ad_next.t-ad.t)>timedelta(hours=24)):
                                add_node(ad)
                            else:
                                add_node(ad)
                                apnd = False
                        elif apnd & (ad_next is None):
                            add_node(ad)
                        elif (not apnd) & (ad_next is not None):
                            if ((ad_next.t-ad.t)>timedelta(hours=24)):
                                apnd = True
                if detected.indices[0][1] != (
                    len(lon_std)-steps_per_day-1):
                    add_node(IK)    
                if i == len(detected.times.IK)-2:                             # if the next drift is the last drift
                    filtered.times.ID.append(starttime+timedelta(hours=detected.indices[i+1][0]*2))                    # keep the next ID
                    ID = Node(satcat,
                            starttime+timedelta(hours=detected.indices[i+1][0]*2),
                            index=detected.indices[i+1][0],
                            ntype='ID',
                            lon=longitudes[detected.indices[i+1][0]],
                            signal="EW")
                    add_node(ID)
                    IK = Node(satcat,
                            detected.times.IK[i+1],
                            index=detected.indices[i+1][1],
                            ntype='IK',
                            lon=longitudes[detected.indices[i+1][1]],
                            signal="EW")
                    apnd = True
                    for j in range(len(detected.times.AD)):
                        ad = Node(satcat,
                            detected.times.AD[j],
                            index=detected.AD_dex[j],
                            ntype="AD",
                            signal="EW")
                        ad_next = Node(satcat,
                            detected.times.AD[j+1],
                            index=detected.AD_dex[j+1],
                            ntype="AD",
                            signal="EW") \
                            if j < (len(detected.times.AD)-1) else None
                        if (ad.t>ID.t)&(ad.t<IK.t):
                            if apnd & (ad_next is not None):
                                if ((ad_next.t-ad.t)>timedelta(
                                    hours=24)):
                                    add_node(ad)
                                else:
                                    add_node(ad)
                                    apnd = False
                            elif apnd & (ad_next is None):
                                add_node(ad)
                            elif (not apnd) & (ad_next is not None):
                                if ((ad_next.t-ad.t)>timedelta(
                                    hours=24)):
                                    apnd = True
                    if detected.indices[i][1] != (
                        len(lon_std)-steps_per_day-1):
                        filtered.times.IK.append(detected.times.IK[i+1])      # keep the next IK
                        add_node(IK)
            else:                                                             # if the next ID and the current IK are 48 hours apart or less
                ID = Node(satcat,
                        detected.times.ID[i],
                        index=detected.indices[i][0],
                        ntype='ID',
                        lon=longitudes[detected.indices[i][0]],
                        signal="EW")                                          # keep the current ID
                add_node(ID)
                AD = Node(satcat,
                        detected.times.IK[i],
                        index=detected.indices[i][1],
                        ntype='AD',
                        lon=longitudes[detected.indices[i][1]],
                        signal="EW")                                          # change the current IK to an AD
                IK = Node(satcat,
                        detected.times.IK[i+1],
                        index=detected.indices[i+1][1],
                        ntype='IK',
                        lon=longitudes[detected.indices[i+1][1]],
                        signal="EW")                                          # exchange the current IK for the next one
                add_node(AD)
                apnd = True
                for j in range(len(detected.times.AD)):
                    ad = Node(satcat,
                      detected.times.AD[j],
                      index=detected.AD_dex[j],
                      ntype="AD",
                      signal="EW")
                    ad_next = Node(satcat,
                      detected.times.AD[j+1],
                      index=detected.AD_dex[j+1],
                      ntype="AD",
                      signal="EW") \
                        if j < (len(detected.times.AD)-1) else None
                    if (ad.t>ID.t)&(ad.t<IK.t):
                        if apnd & (ad_next is not None):
                            if ((ad_next.t-ad.t)>timedelta(hours=24)):
                                add_node(ad)
                            else:
                                add_node(ad)
                                apnd = False
                        elif apnd & (ad_next is None):
                            add_node(ad)
                        elif (not apnd) & (ad_next is not None):
                            if ((ad_next.t-ad.t)>timedelta(hours=24)):
                                apnd = True
                if detected.indices[0][1] != (
                    len(lon_std)-steps_per_day-1):
                    add_node(IK)    
                filtered.times.ID.append(detected.times.ID[i])
                filtered.times.AD.append(detected.times.IK[i])
                filtered.times.IK.append(detected.times.IK[i+1])
                toggle = False                                                # skip the redundant drift
        else:
            toggle = True
    add_node(es)

    ssNS = Node(
            satcat=satcat,
            t=starttime,
            index=0,
            ntype="SS",
            signal="NS",
            mtype="NK")

    for j in range(len(filtered.indices)):
        indices = filtered.indices[j]
        first = True if indices[0] == 0 else False
        times = []
        dexs = []
        inc = inclinations[indices[0]:indices[1]].to_numpy()
        t = np.arange(indices[0],indices[1])*2
        rate = (steps_per_day/(indices[1]-indices[0]))*(np.max(inc)-np.min(inc))
        XIPS_inc_per_day = 0.0005 #0.035/30
        if (rate < XIPS_inc_per_day) and (indices[0] < steps_per_day) and (indices[1] > steps_per_day):
            if filtered.modes.end[j]:
                nodes.append(Node(
                    satcat=satcat,
                    t=starttime+timedelta(hours=indices[1]*2),
                    index=indices[1],
                    ntype="ID",
                    signal="NS",
                    mtype="NK"
                ))
                
            ssNS.mtype = filtered.modes.SK[j]
        elif (rate < XIPS_inc_per_day):
            nodes.append(Node(
                satcat=satcat,
                t=starttime+timedelta(hours=indices[1]*2),
                index=indices[1],
                ntype="IK",
                signal="NS",
                mtype=filtered.modes.SK[j]
            ))
            if filtered.modes.end[j]:
                nodes.append(Node(
                    satcat=satcat,
                    t=starttime+timedelta(hours=indices[1]*2),
                    index=indices[1],
                    ntype="ID",
                    signal="NS",
                    mtype="NK"
                ))
        else:
            dt = [0.0]
            for i in range(len(inc)-1):
                dt.append((inc[i+1]-inc[i])/(2*60*60))
            prev = 1.0

            for i in range(len(dt)-1):
                if np.abs(dt[i])> 5.5e-7:
                    times.append(starttime+timedelta(hours=float(t[i])))
                    dexs.append(i+indices[0])
                    if (np.abs(np.mean(inc[0:i])-np.mean(inc[i:len(inc)]))/(np.std(inc[0:i])+sys.float_info.epsilon))/prev < 1.0:
                        if first and len(times)==2:
                            ssNS.mtype = filtered.modes.SK[0]
                            first = False
                    elif len(times)==2:
                        first = False
                    prev = np.abs(np.mean(inc[0:i])-np.mean(inc[i:len(inc)]))/(np.std(inc[0:i])+sys.float_info.epsilon)

            if len(times)>0:
                nodes.append(Node(
                    satcat=satcat,
                    t=times[0],
                    index=dexs[0],
                    ntype="IK",
                    signal="NS",
                    mtype=filtered.modes.SK[j]
                ))
                ssNS.mtype = "NK"
                if filtered.modes.end[j]:
                    nodes.append(Node(
                        satcat=satcat,
                        t=starttime+timedelta(hours=indices[1]*2),
                        index=indices[1],
                        ntype="ID",
                        signal="NS",
                        mtype="NK"
                    ))
            elif filtered.indices[0][0] == 0:
                ssNS.mtype = filtered.modes.SK[0]
            else:
                ssNS.mtype = "NK"
    nodes.append(ssNS)
    nodes.sort(key=lambda x: x.t)
    
    # Convert timestamp back into timeindex and format the output to the correct format in a pandas dataframe
    ObjectID_list = []
    TimeIndex_list = []
    Direction_list = []
    Node_list = []
    Type_list = []
    for i in range(len(nodes)):
        ObjectID_list.append(nodes[i].satcat)
        TimeIndex_list.append(int(((nodes[i].t-starttime).days*24+(nodes[i].t-starttime).seconds/3600)/2))
        Direction_list.append(nodes[i].signal)
        Node_list.append(nodes[i].type)
        Type_list.append(nodes[i].mtype)
    
    # Initialize data of lists. 
    data = {'ObjectID': ObjectID_list, 
            'TimeIndex': TimeIndex_list,
            'Direction': Direction_list, 
            'Node': Node_list,
            'Type': Type_list} 
    
    # Create the pandas DataFrame 
    prediction_temp = pd.DataFrame(data) 
    frames.append(prediction_temp)

# Create the pandas DataFrame 
prediction_HR = pd.concat(frames)
# prediction_HR.rename(columns={"Node": "Node_HP", "Type": "Type_HP"},inplace=  True)
divisor = 12
prediction_HR['Quotient'], _ = zip(*prediction_HR['TimeIndex'].map(lambda x: divmod(x, divisor)))  

# INPUT/OUTPUT PATHS WITHIN THE DOCKER CONTAINER
# TRAINED_MODEL_DIR = Path('./trained_model/')
# TEST_DATA_DIR = Path('../../../data/phase_1_v3/fold_0/valid')
# TEST_PREDS_FP = Path('../../../data/submission/submission_metric_sensitive_xgboost.csv')


# Rest of configuration, specific to this submission
feature_cols = [
    "Eccentricity",
    "Semimajor Axis (m)",
    "Inclination (deg)",
    "RAAN (deg)",
    "Argument of Periapsis (deg)",
    "True Anomaly (deg)",
    "Latitude (deg)",
    "Longitude (deg)",
    "Altitude (m)",
    "X (m)",
    "Y (m)",
    "Z (m)",
    "Vx (m/s)",
    "Vy (m/s)",
    "Vz (m/s)"
]

lag_steps = 5

test_data, updated_feature_cols = utils.tabularize_data(
    TEST_DATA_DIR, feature_cols, lag_steps=lag_steps)

# Load the trained models (don't use the utils module, use pickle)
model_EW = pickle.load(open(TRAINED_MODEL_DIR / 'XG_model_EW.pkl', 'rb'))
model_NS = pickle.load(open(TRAINED_MODEL_DIR / 'XG_model_NS.pkl', 'rb'))
le_EW = pickle.load(open(TRAINED_MODEL_DIR / 'XG_le_EW.pkl', 'rb'))
le_NS = pickle.load(open(TRAINED_MODEL_DIR / 'XG_le_NS.pkl', 'rb'))

# Make predictions on the test data for EW
test_data['Predicted_EW'] = le_EW.inverse_transform(
    model_EW.predict(test_data[updated_feature_cols])
)

# Make predictions on the test data for NS
test_data['Predicted_NS'] = le_NS.inverse_transform(
    model_NS.predict(test_data[updated_feature_cols])
)
list_unique_ids = test_data["ObjectID"].unique().tolist()
test_results_frames = []
for idx in list_unique_ids:
    try:
        selected = test_data[test_data["ObjectID"]==idx][['TimeIndex', 'ObjectID', 
                    'Predicted_EW', 'Predicted_NS']]
        # Detect the first occurrence of a new value by comparing each row to the next one
        selected['first_occurrence'] = (selected['Predicted_EW'] != selected['Predicted_EW'].shift(1)) | (selected.index == 0)

        # Filter the DataFrame to keep only the first occurrences
        first_occurrences_df_ew = selected[selected['first_occurrence']]

        # Drop the auxiliary column if it's no longer needed
        first_occurrences_df_ew = first_occurrences_df_ew.drop(columns=['first_occurrence'])

        # Now first_occurrences_df contains only the rows where a new sequence starts
        first_occurrences_df_ew = first_occurrences_df_ew[first_occurrences_df_ew["Predicted_EW"]!="XX-XX"][["TimeIndex", "ObjectID", "Predicted_EW"]]    
        # Detect the first occurrence of a new value by comparing each row to the next one
        selected['first_occurrence'] = (selected['Predicted_NS'] != selected['Predicted_NS'].shift(1)) | (selected.index == 0)

        # Filter the DataFrame to keep only the first occurrences
        first_occurrences_df_ns = selected[selected['first_occurrence']]

        # Drop the auxiliary column if it's no longer needed
        first_occurrences_df_ns = first_occurrences_df_ns.drop(columns=['first_occurrence'])

        # Now first_occurrences_df contains only the rows where a new sequence starts
        first_occurrences_df_ns = first_occurrences_df_ns[first_occurrences_df_ns["Predicted_NS"]!="XX-XX"][["TimeIndex", "ObjectID", "Predicted_NS"]]   

        final_df = utils.convert_classifier_output_sep(first_occurrences_df_ew,first_occurrences_df_ns)     

        final_df['Quotient'], _ = zip(*final_df['TimeIndex'].map(lambda x: divmod(x, divisor)))    

        test_results_frames.append(final_df)
    except:
        print(idx)

prediction_XG = pd.concat(test_results_frames)

test_data.drop(columns=["Predicted_EW","Predicted_NS"], inplace=True)

# Load the trained models (don't use the utils module, use pickle)
model_EW = CatBoostClassifier()
model_EW.load_model(TRAINED_MODEL_DIR / 'CB_model_EW')
model_NS = CatBoostClassifier()
model_NS.load_model(TRAINED_MODEL_DIR / 'CB_model_NS')
le_EW = pickle.load(open(TRAINED_MODEL_DIR / 'CB_le_EW.pkl', 'rb'))
le_NS = pickle.load(open(TRAINED_MODEL_DIR / 'CB_le_NS.pkl', 'rb'))

# Make predictions on the test data for EW
test_data['Predicted_EW'] = le_EW.inverse_transform(
    model_EW.predict(test_data[updated_feature_cols])
)

# Make predictions on the test data for NS
test_data['Predicted_NS'] = le_NS.inverse_transform(
    model_NS.predict(test_data[updated_feature_cols])
)

list_unique_ids = test_data["ObjectID"].unique().tolist()
test_results_frames = []
divisor = 12
for idx in list_unique_ids:
    try:
        selected = test_data[test_data["ObjectID"]==idx][['TimeIndex', 'ObjectID', 
                    'Predicted_EW', 'Predicted_NS']]
        # Detect the first occurrence of a new value by comparing each row to the next one
        selected['first_occurrence'] = (selected['Predicted_EW'] != selected['Predicted_EW'].shift(1)) | (selected.index == 0)

        # Filter the DataFrame to keep only the first occurrences
        first_occurrences_df_ew = selected[selected['first_occurrence']]

        # Drop the auxiliary column if it's no longer needed
        first_occurrences_df_ew = first_occurrences_df_ew.drop(columns=['first_occurrence'])

        # Now first_occurrences_df contains only the rows where a new sequence starts
        first_occurrences_df_ew = first_occurrences_df_ew[first_occurrences_df_ew["Predicted_EW"]!="XX-XX"][["TimeIndex", "ObjectID", "Predicted_EW"]]    
        # Detect the first occurrence of a new value by comparing each row to the next one
        selected['first_occurrence'] = (selected['Predicted_NS'] != selected['Predicted_NS'].shift(1)) | (selected.index == 0)

        # Filter the DataFrame to keep only the first occurrences
        first_occurrences_df_ns = selected[selected['first_occurrence']]

        # Drop the auxiliary column if it's no longer needed
        first_occurrences_df_ns = first_occurrences_df_ns.drop(columns=['first_occurrence'])

        # Now first_occurrences_df contains only the rows where a new sequence starts
        first_occurrences_df_ns = first_occurrences_df_ns[first_occurrences_df_ns["Predicted_NS"]!="XX-XX"][["TimeIndex", "ObjectID", "Predicted_NS"]]   

        final_df = utils.convert_classifier_output_sep(first_occurrences_df_ew,first_occurrences_df_ns)     

        final_df['Quotient'], _ = zip(*final_df['TimeIndex'].map(lambda x: divmod(x, divisor)))    

        test_results_frames.append(final_df)
    except:
        print(idx)

prediction_CB = pd.concat(test_results_frames)

test_data.drop(columns=["Predicted_EW","Predicted_NS"], inplace=True)

# Load the trained models (don't use the utils module, use pickle)
model_EW = pickle.load(open(TRAINED_MODEL_DIR / 'RF_model_EW.pkl', 'rb'))
model_NS = pickle.load(open(TRAINED_MODEL_DIR / 'RF_model_NS.pkl', 'rb'))
le_EW = pickle.load(open(TRAINED_MODEL_DIR / 'RF_le_EW.pkl', 'rb'))
le_NS = pickle.load(open(TRAINED_MODEL_DIR / 'RF_le_NS.pkl', 'rb'))

# Make predictions on the test data for EW
test_data['Predicted_EW'] = le_EW.inverse_transform(
    model_EW.predict(test_data[updated_feature_cols])
)

# Make predictions on the test data for NS
test_data['Predicted_NS'] = le_NS.inverse_transform(
    model_NS.predict(test_data[updated_feature_cols])
)

list_unique_ids = test_data["ObjectID"].unique().tolist()
test_results_frames = []
divisor = 12
for idx in list_unique_ids:
    try:
        selected = test_data[test_data["ObjectID"]==idx][['TimeIndex', 'ObjectID', 
                    'Predicted_EW', 'Predicted_NS']]
        # Detect the first occurrence of a new value by comparing each row to the next one
        selected['first_occurrence'] = (selected['Predicted_EW'] != selected['Predicted_EW'].shift(1)) | (selected.index == 0)

        # Filter the DataFrame to keep only the first occurrences
        first_occurrences_df_ew = selected[selected['first_occurrence']]

        # Drop the auxiliary column if it's no longer needed
        first_occurrences_df_ew = first_occurrences_df_ew.drop(columns=['first_occurrence'])

        # Now first_occurrences_df contains only the rows where a new sequence starts
        first_occurrences_df_ew = first_occurrences_df_ew[first_occurrences_df_ew["Predicted_EW"]!="XX-XX"][["TimeIndex", "ObjectID", "Predicted_EW"]]    
        # Detect the first occurrence of a new value by comparing each row to the next one
        selected['first_occurrence'] = (selected['Predicted_NS'] != selected['Predicted_NS'].shift(1)) | (selected.index == 0)

        # Filter the DataFrame to keep only the first occurrences
        first_occurrences_df_ns = selected[selected['first_occurrence']]

        # Drop the auxiliary column if it's no longer needed
        first_occurrences_df_ns = first_occurrences_df_ns.drop(columns=['first_occurrence'])

        # Now first_occurrences_df contains only the rows where a new sequence starts
        first_occurrences_df_ns = first_occurrences_df_ns[first_occurrences_df_ns["Predicted_NS"]!="XX-XX"][["TimeIndex", "ObjectID", "Predicted_NS"]]   

        final_df = utils.convert_classifier_output_sep(first_occurrences_df_ew,first_occurrences_df_ns)     

        final_df['Quotient'], _ = zip(*final_df['TimeIndex'].map(lambda x: divmod(x, divisor)))    

        test_results_frames.append(final_df)
    except:
        print(idx)

prediction_RF = pd.concat(test_results_frames)

resultant_frame = {
                    "ObjectID":	[],
                    "TimeIndex":[],	
                    "Direction": [],	
                    "Node":	[],
                    "Type": []
}
lst_objectid = test_data["ObjectID"].unique().tolist()
for objectid in lst_objectid:
#   print("ObjectID {}".format(objectid))
  selected_HR = prediction_HR[prediction_HR["ObjectID"]==objectid]
  selected_HR["model"] = "HR"
  selected_HR_EW = selected_HR[selected_HR["Direction"]=="EW"]
  selected_HR_NS = selected_HR[selected_HR["Direction"]=="NS"]
  selected_XG = prediction_XG[prediction_XG["ObjectID"]==objectid]
  selected_XG["model"] = "XG"
  selected_XG_EW = selected_XG[selected_XG["Direction"]=="EW"]
  selected_XG_NS = selected_XG[selected_XG["Direction"]=="NS"]
  selected_CB = prediction_CB[prediction_CB["ObjectID"]==objectid]
  selected_CB["model"] = "CB"
  selected_CB_EW = selected_CB[selected_CB["Direction"]=="EW"]
  selected_CB_NS = selected_CB[selected_CB["Direction"]=="NS"]
  selected_RF = prediction_RF[prediction_RF["ObjectID"]==objectid]
  selected_RF["model"] = "RF"
  selected_RF_EW = selected_RF[selected_RF["Direction"]=="EW"]
  selected_RF_NS = selected_RF[selected_RF["Direction"]=="NS"]
  new_frame_EW = []
  new_frame_EW.append(selected_RF_EW)
  new_frame_EW.append(selected_XG_EW)
  new_frame_EW.append(selected_CB_EW)
  new_frame_EW.append(selected_HR_EW)
  new_frame_EW_pd = pd.concat(new_frame_EW)  
  new_frame_NS = []
  new_frame_NS.append(selected_RF_NS)
  new_frame_NS.append(selected_XG_NS)
  new_frame_NS.append(selected_CB_NS)
  new_frame_NS.append(selected_HR_NS)
  new_frame_NS_pd = pd.concat(new_frame_NS)    
  list_quotient_EW = new_frame_EW_pd["Quotient"].unique().tolist()
  list_quotient_NS = new_frame_NS_pd["Quotient"].unique().tolist()
  for quotient_EW in list_quotient_EW:
    #   print("EW unique quotient {}".format(quotient_EW))
      selected_new_frame_EW_pd = new_frame_EW_pd[new_frame_EW_pd["Quotient"]==quotient_EW]
      lst_unique_node = selected_new_frame_EW_pd.Node.unique().tolist()
      lst_unique_type = selected_new_frame_EW_pd.Type.unique().tolist()
      if len(selected_new_frame_EW_pd.model.tolist()) == 1:
        # pass
        if selected_new_frame_EW_pd.model.tolist()[0] == "HR" and min(selected_new_frame_EW_pd.TimeIndex.tolist())==0:
          resultant_frame["ObjectID"].append(selected_new_frame_EW_pd.ObjectID.tolist()[0])
          resultant_frame["TimeIndex"].append(int(np.mean(selected_new_frame_EW_pd.TimeIndex.tolist())))
          resultant_frame["Direction"].append("EW")
          resultant_frame["Node"].append(selected_new_frame_EW_pd.Node.tolist()[0])
          resultant_frame["Type"].append(selected_new_frame_EW_pd.Type.tolist()[0])
      else:
        if len(lst_unique_node)==1 and len(lst_unique_type)==1:
          resultant_frame["ObjectID"].append(selected_new_frame_EW_pd.ObjectID.tolist()[0])
          resultant_frame["TimeIndex"].append(int(np.mean(selected_new_frame_EW_pd.TimeIndex.tolist())))
          resultant_frame["Direction"].append("EW")
          resultant_frame["Node"].append(lst_unique_node[0])
          resultant_frame["Type"].append(lst_unique_type[0])                      
        else:
          if len(list_quotient_EW) ==1:
            resultant_frame["ObjectID"].append(selected_new_frame_EW_pd.ObjectID.tolist()[0])
            resultant_frame["TimeIndex"].append(int(np.mean(selected_new_frame_EW_pd.TimeIndex.tolist())))
            resultant_frame["Direction"].append("EW")   
            resultant_frame["Type"].append(selected_new_frame_EW_pd[selected_new_frame_EW_pd["model"]=="HR"].Type.tolist()[0]) 
            resultant_frame["Node"].append(selected_new_frame_EW_pd[selected_new_frame_EW_pd["model"]=="HR"].Node.tolist()[0])           
          else:
            resultant_frame["ObjectID"].append(selected_new_frame_EW_pd.ObjectID.tolist()[0])
            resultant_frame["TimeIndex"].append(int(np.mean(selected_new_frame_EW_pd.TimeIndex.tolist())))
            resultant_frame["Direction"].append("EW")          
            lst_count = []
            lst_sum = []
            for idx in lst_unique_type:
              lst_sum.append(selected_new_frame_EW_pd[selected_new_frame_EW_pd["Type"]==idx].groupby("TimeIndex").sum().index.tolist()[0])
              lst_count.append(selected_new_frame_EW_pd[selected_new_frame_EW_pd["Type"]==idx].groupby("TimeIndex").count().Type.tolist()[0]) 
            max_value_count = min(lst_count)
            max_indices_count = [index for index, value in enumerate(lst_count) if value == max_value_count]    
            max_value = min(max_indices_count)
            max_indices = [index for index, value in enumerate(max_indices_count) if value == max_value]               
            resultant_frame["Type"].append(lst_unique_type[max_indices[0]])     
            lst_count = []
            lst_sum = []
            for idx in lst_unique_node:
              lst_sum.append(selected_new_frame_EW_pd[selected_new_frame_EW_pd["Node"]==idx].groupby("TimeIndex").sum().index.tolist()[0])
              lst_count.append(selected_new_frame_EW_pd[selected_new_frame_EW_pd["Node"]==idx].groupby("TimeIndex").count().Node.tolist()[0])         
            max_value_count = min(lst_count)
            max_indices_count = [index for index, value in enumerate(lst_count) if value == max_value_count]
            max_value = min(max_indices_count)
            max_indices = [index for index, value in enumerate(max_indices_count) if value == max_value] 
            resultant_frame["Node"].append(lst_unique_node[max_indices[0]])     
  for quotient_NS in list_quotient_NS:
    #   print("NS unique quotient {}".format(quotient_NS))
      selected_new_frame_NS_pd = new_frame_NS_pd[new_frame_NS_pd["Quotient"]==quotient_NS]
      lst_unique_node = selected_new_frame_NS_pd.Node.unique().tolist()
      lst_unique_type = selected_new_frame_NS_pd.Type.unique().tolist()
      if len(selected_new_frame_NS_pd.model.tolist()) == 1:
        # pass
        if selected_new_frame_NS_pd.model.tolist()[0] == "HR" and min(selected_new_frame_NS_pd.TimeIndex.tolist())==0:
          resultant_frame["ObjectID"].append(selected_new_frame_NS_pd.ObjectID.tolist()[0])
          resultant_frame["TimeIndex"].append(int(np.mean(selected_new_frame_NS_pd.TimeIndex.tolist())))
          resultant_frame["Direction"].append("NS")
          resultant_frame["Node"].append(selected_new_frame_NS_pd.Node.tolist()[0])
          resultant_frame["Type"].append(selected_new_frame_NS_pd.Type.tolist()[0])
      else:
        if len(lst_unique_node)==1 and len(lst_unique_type)==1:
          resultant_frame["ObjectID"].append(selected_new_frame_NS_pd.ObjectID.tolist()[0])
          resultant_frame["TimeIndex"].append(int(np.mean(selected_new_frame_NS_pd.TimeIndex.tolist())))
          resultant_frame["Direction"].append("NS")
          resultant_frame["Node"].append(lst_unique_node[0])
          resultant_frame["Type"].append(lst_unique_type[0])                      
        else:
          if len(list_quotient_NS) ==1:
            resultant_frame["ObjectID"].append(selected_new_frame_NS_pd.ObjectID.tolist()[0])
            resultant_frame["TimeIndex"].append(int(np.mean(selected_new_frame_NS_pd.TimeIndex.tolist())))
            resultant_frame["Direction"].append("NS")   
            resultant_frame["Type"].append(selected_new_frame_NS_pd[selected_new_frame_NS_pd["model"]=="HR"].Type.tolist()[0]) 
            resultant_frame["Node"].append(selected_new_frame_NS_pd[selected_new_frame_NS_pd["model"]=="HR"].Node.tolist()[0])           
          else:          
            resultant_frame["ObjectID"].append(selected_new_frame_NS_pd.ObjectID.tolist()[0])
            resultant_frame["TimeIndex"].append(int(np.mean(selected_new_frame_NS_pd.TimeIndex.tolist())))
            resultant_frame["Direction"].append("NS")          
            lst_count = []
            lst_sum = []
            for idx in lst_unique_type:
              lst_sum.append(selected_new_frame_NS_pd[selected_new_frame_NS_pd["Type"]==idx].groupby("TimeIndex").sum().index.tolist()[0])
              lst_count.append(selected_new_frame_NS_pd[selected_new_frame_NS_pd["Type"]==idx].groupby("TimeIndex").count().Type.tolist()[0]) 
            max_value_count = min(lst_count)
            max_indices_count = [index for index, value in enumerate(lst_count) if value == max_value_count]    
            max_value = min(max_indices_count)
            max_indices = [index for index, value in enumerate(max_indices_count) if value == max_value]               
            resultant_frame["Type"].append(lst_unique_type[max_indices[0]])     
            lst_count = []
            lst_sum = []
            for idx in lst_unique_node:
              lst_sum.append(selected_new_frame_NS_pd[selected_new_frame_NS_pd["Node"]==idx].groupby("TimeIndex").sum().index.tolist()[0])
              lst_count.append(selected_new_frame_NS_pd[selected_new_frame_NS_pd["Node"]==idx].groupby("TimeIndex").count().Node.tolist()[0])         
            max_value_count = min(lst_count)
            max_indices_count = [index for index, value in enumerate(lst_count) if value == max_value_count]
            max_value = min(max_indices_count)
            max_indices = [index for index, value in enumerate(max_indices_count) if value == max_value] 
            resultant_frame["Node"].append(lst_unique_node[max_indices[0]])      
prediction = pd.DataFrame(resultant_frame)
# Save the prediction into a csv file 
prediction.to_csv(TEST_PREDS_FP, index=False)  
print("Saved predictions to: {}".format(TEST_PREDS_FP))
time.sleep(360) # TEMPORARY FIX TO OVERCOME EVALAI BUG