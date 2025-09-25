import time
import folium.map
import googlemaps
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
import os.path
from scipy.spatial import ConvexHull
import requests
from haversine import haversine, Unit
from matplotlib.path import Path
import csv
import streamlit as st
import folium
from streamlit_folium import st_folium
from streamlit.components.v1 import html
import ast
from itertools import groupby, permutations
import googlemaps


st.set_page_config(layout="wide")

speed=50
maxhr=11


if 'depot_selection' not in st.session_state:
    st.session_state['depot_selection'] = None

if 'uploaded_flag' not in st.session_state:
    st.session_state['uploaded_flag'] = 0

if 'AllCusts_edited' not in st.session_state:
    st.session_state['AllCusts_edited'] = None    

if 'OdralData_edited' not in st.session_state:
    st.session_state['OdralData_edited'] = None

if 'OdralData_df' not in st.session_state:
    st.session_state['OdralData_df'] = None

if 'TruckData_edited' not in st.session_state:
    st.session_state['TruckData_edited'] = None

if 'TruckData_df' not in st.session_state:
    st.session_state['TruckData_df'] = None

if 'ClusterData_edited' not in st.session_state:
    st.session_state['ClusterData_edited'] = None

if 'ClusterData_df' not in st.session_state:
    st.session_state['ClusterData_df'] = None    

if 'DepotData_edited' not in st.session_state:
    st.session_state['DepotData_edited'] = None

if 'DepotData_df' not in st.session_state:
    st.session_state['DepotData_df'] = None

if 'cluster_list' not in st.session_state:
    st.session_state['cluster_list'] = []

if "cluster_df" not in st.session_state:  # Coordinates of the drawn or given (in the original input) for clusters
    st.session_state["cluster_df"] = pd.DataFrame(columns=["ClusterName", "Features"])     


with st.expander("File Upload",expanded=True):
    col1,col2,col3,col4,col5,col6,col7,col8=st.columns([1,0.5,2,0.5,1,0.5,1,0.5],vertical_alignment="top",gap="small")
    
    col1.text('Step1: Reset')
    reset_button=col1.button("Reset Application",type="primary")
    if reset_button:
        st.cache_data.clear()
        st.cache_resource.clear()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.toast("Application reset!", icon="✅")            


    col3.text('Step2: Upload')
    uploaded_file = col3.file_uploader("", type=["xlsx", "xls"],label_visibility="collapsed",)

    col5.text('Step3: Open')
    run_button = col5.button("Open Uploaded File", type="primary")
    if run_button:
            st.session_state['uploaded_flag'] = True  # Set a flag in session_state
            st.rerun()

    DepotData = st.session_state.get("DepotData")
    depot_options = ['Select'] + list(DepotData['DEPOT'].unique()) if DepotData is not None else ['Select']
    col7.text('Step4: Depot')
    selected_depot = col7.selectbox("Select Depot", options=depot_options, index=0, key="depotselection",label_visibility="collapsed")
    st.session_state['depot_selection'] = selected_depot

    if uploaded_file is not None:
        try:
            # Read all sheets into a dictionary of DataFrames
            sheets_dict = pd.read_excel(uploaded_file, sheet_name=None)
            # Store each sheet in session_state
            for sheet_name, df in sheets_dict.items():
                st.session_state[f"{sheet_name}"] = df
        except Exception as e:
            st.error(f"Error reading the file: {e}")

        # DYNAMIC FILE LOGIC #
        # ----------------------------------------------------------------------------------------
        # OdralData / ClusterData / TruckData / DepotData => original data coming in the file stored in session state
        # OdralData_edited / ClusterData_edited / TruckData_edited / DepotData_edited => edited data by user stored in sessionstate
        # OdralData_df / ClusterData_df / TruckData_df / DepotData_df => if there is an edited data then OdralData_edited else OdralData (original)
        # ----------------------------------------------------------------------------------------

if st.session_state.get('uploaded_flag', 0) == True and st.session_state.get('depot_selection', None) is not None:

    with st.expander("Customer Data (Editable for Simulation)", expanded=True):

        # All customers
        if (st.session_state.AllCusts_edited is not None and not st.session_state.AllCusts_edited.empty):
            st.session_state.AllCusts_df = st.session_state.AllCusts_edited
        else:
            st.session_state.AllCusts_df = st.session_state.get("OdralData").copy() # original data coming in the file

        with st.form("Customer_Data_form"):
            AllCusts_edited = st.data_editor(st.session_state.AllCusts_df.sort_values(by="Demand", ascending=False), width='stretch', hide_index=True)
            submitted = st.form_submit_button("Save Customer Data", type="primary")
            if submitted:              
                st.session_state.AllCusts_edited = AllCusts_edited
                st.rerun()
                st.toast("Customer data saved!", icon="✅")

    # Filtered Customers by depot
    OdralData=st.session_state.AllCusts_df # original data coming in the file
    st.session_state.OdralData_df = OdralData[OdralData['DEPOT'] == st.session_state.depot_selection].copy() if OdralData is not None and st.session_state.depot_selection is not None and 'DEPOT' in OdralData.columns else pd.DataFrame()



    with st.expander("Cluster Mapping", expanded=True):
        col1,col2=st.columns([4,2])
        # ------------------------------------------------------------------------------------------------------------------------
        # Prepare the map with customers and drawing tools
        # ------------------------------------------------------------------------------------------------------------------------
        
        ClusterData=st.session_state.get("ClusterData") # original data coming in the file
        if (
            st.session_state.ClusterData_edited is not None and
            st.session_state.depot_selection is not None and
            'DEPOT' in st.session_state.ClusterData_edited.columns and
            not st.session_state.ClusterData_edited[st.session_state.ClusterData_edited['DEPOT'] == st.session_state.depot_selection].empty
        ):
            st.session_state.ClusterData_df = st.session_state.ClusterData_edited
        else:
            st.session_state.ClusterData_df = ClusterData[ClusterData['DEPOT'] == st.session_state.depot_selection].copy() if ClusterData is not None and st.session_state.depot_selection is not None and 'DEPOT' in ClusterData.columns else pd.DataFrame()

        if st.session_state.ClusterData_df is not None and st.session_state.depot_selection:
            st.session_state.cluster_list = st.session_state.ClusterData_df[st.session_state.ClusterData_df['DEPOT'] == st.session_state.depot_selection]['ClusterName'].tolist()
        else:
            st.session_state.cluster_list = []

        if st.session_state.AllCusts_df is not None:
            # Ensure required columns exist
            required_columns = {'LAT', 'LON', 'DEPOT'}
            if not required_columns.issubset(st.session_state.AllCusts_df.columns):
                st.error(f"Missing columns in data: {required_columns - set(st.session_state.AllCusts_df.columns)}")
            else:
                # Create a folium map centered on the mean location
                center_lat = 38.821866877802435
                center_lon = 33.28993733084615
                m = folium.Map(location=[center_lat, center_lon], zoom_start=6)

                # Assign a color to each unique DEPOT
                depots = st.session_state.AllCusts_df['DEPOT'].unique()
                colors = px.colors.qualitative.Plotly
                depot_color_map = {depot: colors[i % len(colors)] for i, depot in enumerate(depots)}

                # Add customer markers
                for _, row in st.session_state.AllCusts_df.iterrows():
                    popup_html = f"""
                    <b>Depot:</b> {row['DEPOT']}<br>
                    <b>CustNo:</b> {row['CUST_NO']}<br>
                    <b>Cust:</b> {row['CUST_NAME']}<br>
                    <b>Type:</b> {row['CUST_TYPE']}<br>
                    <b>MapRef:</b> {row['MAP_REF']}<br>
                    <b>City:</b> {row['CITY']}<br>
                    <b>Demand:</b> {row['Demand']}
                    """
                    folium.CircleMarker(
                        location=[row['LAT'], row['LON']],
                        radius=6,
                        color=depot_color_map[row['DEPOT']],
                        fill=True,
                        fill_color=depot_color_map[row['DEPOT']],
                        fill_opacity=0.7,
                        popup=folium.Popup(popup_html, max_width=300, min_width=150)
                    ).add_to(m)

                # Draw polygons from cluster_df["Features"]
                for _, row in st.session_state.cluster_df.iterrows():
                    try:
                        features = row["Features"]
                        if isinstance(features, str):
                            idx = features.find("[")
                            if idx != -1:
                                features = features[idx:]
                                features = features.replace("}", "")
                        if pd.notna(features) and features != "":
                            coords = ast.literal_eval(features)
                            # Accept both [[lon, lat], ...] and [[[lon, lat], ...]]
                            if isinstance(coords[0][0], (float, int)):
                                polygon_coords = [[lat, lon] for lon, lat in coords]
                            else:
                                polygon_coords = [[lat, lon] for lon, lat in coords[0]]
                            folium.Polygon(
                                locations=polygon_coords,
                                color="blue",
                                fill=True,
                                fill_opacity=0.2,
                                popup=row["ClusterName"]
                            ).add_to(m)
                    except Exception as e:
                        continue

                # Add drawing tools (rectangle and polygon) to the map
                draw = Draw(
                    export=False,
                    draw_options={"polyline": False, "circle": False,"marker": False,"circlemarker": False,"rectangle": True,"polygon": True},
                    edit_options={"edit": True}
                )
                draw.add_to(m)

                # Display the map in Streamlit using HTML component
                with col1:
                    map_html = m._repr_html_()
                    html(map_html, height=650, width=1200)
                

        # ------------------------------------------------------------------------------------------------------------------------
        # Get Coordinates of the clusters from HTML popup and save to DataFrame
        # ------------------------------------------------------------------------------------------------------------------------
        
        with col2.form("save_clusters_form"):

            # Capture origincal ClusterData from the file if Features not empty
            if st.session_state.ClusterData_df is not None and not st.session_state.ClusterData_df.empty and st.session_state.ClusterData_df['Features'].notna().any():
                st.session_state.cluster_df = st.session_state.ClusterData_df[['ClusterName', 'Features']].copy()
            elif st.session_state.ClusterData_edited is not None and not st.session_state.ClusterData_edited.empty and st.session_state.ClusterData_edited['Features'].notna().any():
                st.session_state.cluster_df = st.session_state.ClusterData_edited[['ClusterName', 'Features']].copy()
            else:
                st.session_state.cluster_df = pd.DataFrame(columns=['ClusterName', 'Features'])

            st.session_state.cluster_df = st.data_editor(st.session_state.cluster_df.reset_index(drop=True), num_rows="dynamic", width='content', height=550, column_config={"ClusterName": st.column_config.SelectboxColumn("ClusterName", options=st.session_state.cluster_list)})

            def get_cluster_name(lat, lon, cluster_df):
                for idx, row in cluster_df.iterrows():
                    try:
                        features = row["Features"]
                        if isinstance(features, str):
                            idx = features.find("[")
                            if idx != -1:
                                features = features[idx:]
                                features = features.replace("}", "")
                        if pd.notna(features) and features != "":
                            coords = ast.literal_eval(features)
                            # Accept both [[lon, lat], ...] and [[[lon, lat], ...]]
                        if isinstance(coords[0][0], (float, int)):
                            polygon = [(lat, lon) for lon, lat in coords]
                        else:
                            polygon = [(lat, lon) for lon, lat in coords[0]]
                        path = Path(polygon)
                        if path.contains_point((lat, lon)):
                            return row["ClusterName"]
                    except Exception:
                        continue
                return None
            
            col1,col2,col3=st.columns([1,1,1])
            # Assign customers to clusters based on their coordinates
            submitted = col2.form_submit_button("Save Clusters",type="primary")

            if submitted:
                st.toast("Clusters saved!", icon="✅") # A FORM IS USED JUST TO KEEP THIS PART INACTIVE DURING CLUSTER DRAWING AND FORM REQUIRES A BUTTON 

            

    TruckData=st.session_state.get("TruckData")
    if (
        st.session_state.TruckData_edited is not None and
        st.session_state.depot_selection is not None and
        not st.session_state.TruckData_edited[st.session_state.TruckData_edited['DEPOT'] == st.session_state.depot_selection].empty
    ):
        st.session_state.TruckData_df = st.session_state.TruckData_edited
    else:
        st.session_state.TruckData_df = TruckData[TruckData['DEPOT'] == st.session_state.depot_selection].copy() if TruckData is not None and st.session_state.depot_selection is not None else pd.DataFrame()

    DepotData=st.session_state.get("DepotData")
    if (
        st.session_state.DepotData_edited is not None and
        st.session_state.depot_selection is not None and
        not st.session_state.DepotData_edited[st.session_state.DepotData_edited['DEPOT'] == st.session_state.depot_selection].empty
    ):
        st.session_state.DepotData_df = st.session_state.DepotData_edited
    else:
        st.session_state.DepotData_df = DepotData[DepotData['DEPOT'] == st.session_state.depot_selection].copy() if DepotData is not None and st.session_state.depot_selection is not None else pd.DataFrame()

    with st.expander("Cluster Data (Editable for Simulation)", expanded=True):
        with st.form("cluster_data_form"):
            st.session_state.ClusterData_edited = st.data_editor(st.session_state.ClusterData_df, width='content', hide_index=True,num_rows="dynamic")
            submitted = st.form_submit_button("Save Cluster Data", type="primary")
            if submitted:
                st.session_state.ClusterData_df = st.session_state.ClusterData_edited
                st.toast("Cluster data saved!", icon="✅")
                
    with st.expander("Truck Data (Editable for Simulation)", expanded=True):
        with st.form("truck_data_form"):
            st.session_state.TruckData_edited = st.data_editor(st.session_state.TruckData_df, width='content', hide_index=True,num_rows="dynamic")
            submitted = st.form_submit_button("Save Truck Data", type="primary")
            if submitted:
                st.session_state.TruckData_df = st.session_state.TruckData_edited
                st.toast("Truck data saved!", icon="✅")

    with st.expander("Depot Data (Editable for Simulation)", expanded=True):
        with st.form("depot_data_form"):
            st.session_state.DepotData_edited = st.data_editor(st.session_state.DepotData_df, width='content', hide_index=True)
            submitted = st.form_submit_button("Save Depot Data", type="primary")
            if submitted:
                st.session_state.DepotData_df = st.session_state.DepotData_edited
                st.toast("Depot data saved!", icon="✅")


    run_plan = st.button("Run Planning",type="primary",width="stretch")
    if run_plan:

        if st.session_state.OdralData_df is not None and st.session_state.cluster_df is not None and not st.session_state.cluster_df.empty:
            st.session_state.OdralData_df["ClusterName"] = st.session_state.OdralData_df.apply( lambda row: get_cluster_name(row["LAT"], row["LON"], st.session_state.cluster_df), axis=1)
    
        with st.expander("Clustered Customers Summary",expanded=True):
            if st.session_state.OdralData_df is not None and 'ClusterName' in st.session_state.OdralData_df.columns:
                summary = st.session_state.OdralData_df.groupby('ClusterName').agg({'CUST_NO': 'count', 'Demand': 'sum'}).reset_index().rename(columns={'CUST_NO': 'CustNb'})
                summary['Demand'] = np.ceil(summary['Demand'])
                # Add a total row to summary
                total_row = {
                    "ClusterName": "Total",
                    "CustNb": summary["CustNb"].sum(),
                    "Demand": summary["Demand"].sum()
                }
                summary = pd.concat([summary, pd.DataFrame([total_row])], ignore_index=True)
                col1,col2=st.columns([2,7])
                col1.dataframe(summary.sort_values(by="Demand", ascending=False), width='content', height=500, hide_index=True)
                col2.dataframe(st.session_state.OdralData_df.sort_values(by="ClusterName"), width='content', height=500, hide_index=True)
            else:
                st.info("No clustered customers to summarize. Please define and save clusters first.")


        # #------------------------------------------------------------------------------------------------------------------------
        # # Daily Trip Plan 
        # #------------------------------------------------------------------------------------------------------------------------

        if st.session_state.ClusterData_df is not None:
            days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
            TripPlan = st.session_state.ClusterData_df.copy()

            TripPlan.drop(columns=['DEPOT', 'Features'], errors='ignore', inplace=True)
            for day in days:
                new_col = f"Trip{day}"
                if 'CombineID' in st.session_state.ClusterData_df.columns:
                    # For each CombineID+day, join ClusterNames with '+'
                    trip_map = (
                        TripPlan[TripPlan[day].notna() & (TripPlan[day] != '')]
                        .groupby('CombineID')['ClusterName']
                        .apply(lambda x: '+'.join(sorted(x)))
                    )
                    def trip_value(row):
                        if pd.notna(row[day]) and row[day] != '':
                            if pd.isna(row['CombineID']) or row['CombineID'] == '':
                                return row['ClusterName']
                            else:
                                return trip_map.get(row['CombineID'], '')
                        else:
                            return ''
                    TripPlan[new_col] = TripPlan.apply(trip_value, axis=1)
                else:
                    # If no CombineID, just use ClusterName if day is not None/empty
                    TripPlan[new_col] = TripPlan.apply(
                        lambda row: row['ClusterName'] if pd.notna(row[day]) and row[day] != '' else '', axis=1
                    )
            # TripPlan # CONTROL QUERY

            #------------------------------------------------------------------------------------------------------------------------
            # List of unique trips in TripPlan
            #------------------------------------------------------------------------------------------------------------------------

            # Get unique values in TripDay columns
            trip_day_cols = [col for col in TripPlan.columns if col.startswith("Trip")]
            trip_list = set()
            for col in trip_day_cols:
                trip_list.update(TripPlan[col].unique())
            trip_list = {v for v in trip_list if pd.notna(v) and v != ''}

            # Prepare TripPlan by removing duplicates based on TripMon to TripSat combination
            # Remove duplicate trip values for each day, keeping only the first occurrence
            for day in days:
                trip_col = f"Trip{day}"
                if trip_col in TripPlan.columns:
                    # Find duplicated trip values (excluding blanks)
                    mask = (TripPlan[trip_col] != '') & TripPlan[trip_col].duplicated(keep='first')
                    TripPlan.loc[mask, trip_col] = ''
            # Create a column 'unique' that contains the unique non-empty value from TripMon to TripSat for each row
            trip_day_cols = [f"Trip{day}" for day in days]
            TripPlan['unique'] = TripPlan[trip_day_cols].apply(
                lambda row: ','.join(sorted({v for v in row if pd.notna(v) and v != ''})), axis=1
            )
            TripPlan['ClusterName'] = TripPlan.apply(lambda row: row['unique'] if ',' not in str(row['unique']) and str(row['unique']) != '' else row.get('ClusterName', ''), axis=1)
            TripPlan = TripPlan[TripPlan['unique'] != ''].reset_index(drop=True)
            TripPlan = TripPlan.drop(columns=['unique'])
            # TripPlan # CONTROL QUERY

            # Prepare ClusterData for different trip plans

            for day in days:
                if day in st.session_state.ClusterData_df.columns:
                    st.session_state.ClusterData_df[day].fillna(0, inplace=True)

            # Set default values for specific columns if they are None or missing
            for col in ['Traffic', 'Driver', 'Divide', 'Demand_Multiplier', 'CustNb_Multiplier']:
                if col in st.session_state.ClusterData_df.columns:
                    st.session_state.ClusterData_df[col] = st.session_state.ClusterData_df[col].fillna(1)
                else:
                    ClusterData[col] = 1
            st.session_state.ClusterData_df['Trip/Wk'] = st.session_state.ClusterData_df[['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']].astype(float).sum(axis=1)

            # ------------------------------------------------------------------------------------------------------------------------
            
            
            required_cols = ['ClusterName', 'Demand_Multiplier', 'CustNb_Multiplier', 'Traffic', 'Driver', 'Divide', 'Trip/Wk']
            missing_cols = [col for col in required_cols if col not in st.session_state.OdralData_df.columns]
            if missing_cols:
                # Merge OdralData with ClusterData to get Trip/Wk by joining on ClusterName
                st.session_state.OdralData_df = st.session_state.OdralData_df.merge( st.session_state.ClusterData_df[['ClusterName', 'Demand_Multiplier','CustNb_Multiplier','Traffic','Driver','Divide','Trip/Wk']], on='ClusterName', how='left')

            st.session_state.OdralData_df['Cyl/Trip'] = (st.session_state.OdralData_df['ANNUAL_DEMAND'] * st.session_state.OdralData_df['Demand_Multiplier']) / (st.session_state.OdralData_df['Trip/Wk'] * 48 * st.session_state.OdralData_df['Divide'].fillna(1))
            st.session_state.OdralData_df['Drop/Trip'] = (st.session_state.OdralData_df['ANNUAL_DROP'] * st.session_state.OdralData_df['CustNb_Multiplier']) / (st.session_state.OdralData_df['Trip/Wk'] * 48 * st.session_state.OdralData_df['Divide'].fillna(1))

        # Filter OdralData for each element in trip_list where 'ClusterName' exists in trip_list
        if st.session_state.OdralData_df is not None and 'ClusterName' in st.session_state.OdralData_df.columns and trip_list:
            # Prepare a list to collect trip results
            trip_results = []
            for trip in trip_list:
                trip_str = str(trip)
                filtered = st.session_state.OdralData_df[st.session_state.OdralData_df['ClusterName'].isin(trip_str.split('+'))]
                filtered['FCluster'] = trip_str
                # Find up to 8 bounding customers for this trip using ConvexHull
                bounding_customers = filtered
                if not filtered.empty and len(filtered) >= 3:
                    try:
                        points = filtered[["LAT", "LON"]].values
                        hull = ConvexHull(points)
                        hull_indices = hull.vertices
                        # If more than 8, select 8 that are most spread out (evenly spaced on hull)
                        if len(hull_indices) > 8:
                            step = max(1, len(hull_indices) // 8)
                            selected_indices = hull_indices[::step][:8]
                        else:
                            selected_indices = hull_indices
                        bounding_customers = filtered.iloc[selected_indices]
                    except Exception as e:
                        st.warning(f"ConvexHull failed for trip {trip}: {e}")

                # Calculate shortest route visiting all bounding_customers, starting and ending at Depot
                if not filtered.empty:
                    depot_row = DepotData[DepotData['DEPOT'] == st.session_state.depot_selection]
                    if not depot_row.empty:
                        depot_lat = depot_row.iloc[0]['LAT']
                        depot_lon = depot_row.iloc[0]['LON']
                        customer_coords = bounding_customers[["LAT", "LON"]].values.tolist()
                        min_distance = float('inf')
                        best_route = None
                        for perm in permutations(customer_coords):
                            route = [(depot_lat, depot_lon)] + list(perm) + [(depot_lat, depot_lon)]
                            total_dist = sum(
                                haversine(route[i], route[i+1], unit=Unit.KILOMETERS)
                                for i in range(len(route)-1)
                            )
                            if total_dist < min_distance:
                                min_distance = total_dist
                                best_route = route
                        trip_cyl_sum = filtered['Cyl/Trip'].sum().round(1)
                        trip_drop_sum = filtered['Drop/Trip'].sum().round(1)
                        trip_driver_max=filtered['Driver'].max()
                        trip_traffic=filtered['Traffic'].mean().round(1)

                        # Calculate Google Maps driving distance for best_route and assign to TripKm
    
                        gmaps = googlemaps.Client(key=st.secrets["gmapsapi"])
                        google_trip_km = None
                        if gmaps and best_route and len(best_route) > 1:
                            try:
                                # Prepare waypoints for Google Maps Directions API
                                origin = f"{best_route[0][0]},{best_route[0][1]}"
                                destination = f"{best_route[-1][0]},{best_route[-1][1]}"
                                waypoints = "|".join([f"{lat},{lon}" for lat, lon in best_route[1:-1]])
                                url = (
                                    f"https://maps.googleapis.com/maps/api/directions/json?"
                                    f"origin={origin}&destination={destination}"
                                )
                                
                                if waypoints:
                                    url += f"&waypoints={waypoints}"
                                url += f"&key={st.secrets["gmapsapi"]}"
                                response = requests.get(url)
                                if response.status_code == 200:
                                    data = response.json()
                                    if data["status"] == "OK":
                                        total_meters = sum(leg["distance"]["value"] for route in data["routes"] for leg in route["legs"])
                                        google_trip_km = np.ceil(total_meters / 1000)
                            except Exception as e:
                                google_trip_km = None

                        trip_km_value = google_trip_km if google_trip_km is not None else np.ceil(min_distance * 1.3)
                        trip_results.append({
                            "Trip": trip,
                            "TripKm": trip_km_value,
                            "Cyl/Trip": trip_cyl_sum,
                            "Drop/Trip": trip_drop_sum,                        
                            "Driver": trip_driver_max,
                            "Traffic": trip_traffic,
                            "Route": best_route,                        
                        })

            # Show trip km and routing as a dataframe
            if trip_results:
                trip_data_df = pd.DataFrame(trip_results)    
            trip_data_df['Drop/Trip'] = np.ceil(trip_data_df['Drop/Trip'])

            if trip_data_df is None or trip_data_df.empty:
                st.info("No trips to display. Please ensure clusters and customers are defined.")
            else:
                with st.expander("Trips on the Map", expanded=True):

                    col1, col2 = st.columns([1, 2])
                #    ROUTES ARE HERE :) USE AI TO DRAW ON THE MAP

                    col1.dataframe(trip_data_df[['Trip', 'TripKm', 'Cyl/Trip', 'Drop/Trip', 'Driver', 'Traffic']], width='content', height=500, hide_index=True)
                    # Ask user to select a Trip and show the route on the map
                    if not trip_data_df.empty:
                        trip_options = trip_data_df['Trip'].tolist()
                        # Use session state to avoid rerun on trip selection
                        if "selected_trip" not in st.session_state:
                            st.session_state.selected_trip = trip_options[0] if trip_options else None
                        selected_trip = col2.selectbox(
                            "Select a Trip to view its route",
                            trip_options,
                            key="selected_trip_selectbox",
                            index=trip_options.index(st.session_state.selected_trip) if st.session_state.selected_trip in trip_options else 0
                        )
                        st.session_state.selected_trip = selected_trip
                        selected_route = trip_data_df.loc[trip_data_df['Trip'] == selected_trip, 'Route'].values[0]
                        if selected_route:
                            depot_lat, depot_lon = selected_route[0]
                            route_map = folium.Map(location=[depot_lat, depot_lon], zoom_start=7)
                            folium.PolyLine(selected_route, color="red", weight=4, opacity=0.8).add_to(route_map)
                            for idx, (lat, lon) in enumerate(selected_route):
                                if idx == 0 or idx == len(selected_route) - 1:
                                    folium.Marker([lat, lon], popup="Depot", icon=folium.Icon(color="blue")).add_to(route_map)
                                else:
                                    folium.Marker([lat, lon], popup=f"Stop {idx}", icon=folium.Icon(color="green")).add_to(route_map)
                            with col2:
                                html_map = route_map._repr_html_()
                                html(html_map, height=400, width=1000)

                # Join TripKm, Cyl/Trip, Drop/Trip and Traffic from trip_data_df to TripPlan for each day
                for col_prefix, trip_col in [
                    ("Km", "TripKm"),
                    ( "Drop", "Drop/Trip"),
                    ( "Traffic", "Traffic"),
                    ( "Cyl", "Cyl/Trip"),            
                ]:
                    for day in days:
                        trip_day_col = f"Trip{day}"
                        out_col = f"{day}{col_prefix}"
                        if trip_day_col in TripPlan.columns:
                            TripPlan[out_col] = TripPlan[trip_day_col].map(
                                trip_data_df.set_index("Trip")[trip_col] if not trip_data_df.empty else {}
                            )
                

                # Calculate hours for each day using the formula for each day's Km, Drop, Cyl and Traffic
                for day in days:
                    km_col = f"{day}Km"
                    drop_col = f"{day}Drop"
                    cyl_col = f"{day}Cyl"
                    traffic_col = f"{day}Traffic"
                    hr_col = f"{day}Hr"
                    if all(col in TripPlan.columns for col in [km_col, drop_col, cyl_col, traffic_col]):
                        TripPlan[hr_col] = (
                            (TripPlan[km_col] * TripPlan[traffic_col] / (speed * TripPlan['Driver'])) +
                            (TripPlan[drop_col] * 0.5) +
                            (TripPlan[cyl_col] * 2 * 20 / 3600)
                        ).round(1)
                
                # Replace NaN or blank values in MonCyl, TueCyl, ..., MonHr, TueHr with 0
                for day in days:
                    cyl_col = f"{day}Cyl"
                    hr_col = f"{day}Hr"
                    if cyl_col in TripPlan.columns:
                        TripPlan[cyl_col] = TripPlan[cyl_col].replace('', 0).fillna(0)
                    if hr_col in TripPlan.columns:
                        TripPlan[hr_col] = TripPlan[hr_col].replace('', 0).fillna(0)
                
                TripPlan['MaxCyl']=TripPlan[[f"{day}Cyl" for day in days]].max(axis=1).round(1)
                TripPlan = TripPlan.sort_values(by='MaxCyl', ascending=False).reset_index(drop=True)

                TruckData=st.session_state.TruckData_edited
                TruckData=TruckData[TruckData['DEPOT']==st.session_state.depot_selection] if TruckData is not None and st.session_state.depot_selection else None

                #----------------------------------------------------------------
                # TRUCK ASSIGNMENT
                #----------------------------------------------------------------

                # Assign truck for each day in TripPlan based on capacity and hour constraints
                # Start assigning from the smallest capacity truck
                for day in days:
                    cyl_col = f"{day}Cyl"
                    hr_col = f"{day}Hr"
                    truck_col = f"{day}Truck"
                    TripPlan[truck_col] = None
                    if TruckData is not None and not TruckData.empty:
                        # Sort trucks by ascending capacity
                        sorted_trucks = TruckData.sort_values(by="CAPACITY")
                        # Track total hours and cylinders assigned per truck for this day
                        truck_usage = {truck: {"hr": 0, "cyl": 0} for truck in sorted_trucks['TRUCK']}
                        for idx, row in TripPlan.iterrows():
                            # Skip assignment if cylinder count is 0 or missing
                            if row[cyl_col] == 0 or pd.isna(row[cyl_col]):
                                TripPlan.at[idx, truck_col] = None
                                continue
                            assigned = False
                            for _, truck_row in sorted_trucks.iterrows():
                                truck = truck_row['TRUCK']
                                cap = truck_row['CAPACITY']
                                total_hr = truck_usage[truck]["hr"] + row[hr_col]
                                total_cyl = truck_usage[truck]["cyl"] + row[cyl_col]
                                # If total_hr <= 5, allow the same truck to be used in multiple trips in the same day
                                if row[hr_col] <= 5 and total_hr <= maxhr and cap >= row[cyl_col]:
                                    TripPlan.at[idx, truck_col] = truck
                                    # Do not accumulate cylinders for this trip (reset to 0 for next trip)
                                    truck_usage[truck]["hr"] = total_hr
                                    truck_usage[truck]["cyl"] = 0
                                    assigned = True
                                    break
                                elif total_hr <= maxhr and total_cyl <= cap and row[hr_col] < maxhr and cap >= row[cyl_col]:
                                    TripPlan.at[idx, truck_col] = truck
                                    truck_usage[truck]["hr"] = total_hr
                                    truck_usage[truck]["cyl"] = total_cyl
                                    assigned = True
                                    break
                            if not assigned:
                                # If not assigned and cyl > 0, put a warning icon
                                TripPlan.at[idx, truck_col] = "⚠️"

                # Check for infeasible solution (⚠️ in any truck column)
                if any(TripPlan[f"{day}Truck"].astype(str).str.contains("⚠️").any() for day in days):
                    st.toast("Infeasible Solution", icon="⚠️")
                else:
                    st.toast("Feasible Solution", icon="✅")

                with st.expander("Daily Trip Plan", expanded=True):
                    # REPORT -----------------------------------------------------------------
                    TripPlan_Output=TripPlan.copy()
                    TripPlan_Output = TripPlan_Output[[f'Trip{day}' for day in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]] + [f'{day}Truck' for day in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat",]] + [f'{day}Cyl' for day in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat",]] + [f'{day}Hr' for day in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat",]]]
                    # Round up {day}Cyl values to nearest integer for each day
                    for day in days:
                        cyl_col = f"{day}Cyl"
                        if cyl_col in TripPlan_Output.columns:
                            TripPlan_Output[cyl_col] = np.ceil(TripPlan_Output[cyl_col])

                    st.dataframe(TripPlan_Output, width='content',height=500,hide_index=True) 
                #-----------------------------------------------------------------------            
                
                #----------------------------------------------------------------
                # KPI
                #----------------------------------------------------------------

                with st.expander("Truck Utilization Summary", expanded=True):
                    # Calculate and display utilization ratio for each truck per day (cylinders and hours)
                    if TruckData is not None and not TruckData.empty:
                        # Calculate utilization ratios
                        cyl_utilization = {}
                        hr_utilization = {}
                        for day in days:
                            truck_col = f"{day}Truck"
                            cyl_col = f"{day}Cyl"
                            hr_col = f"{day}Hr"
                            for _, truck_row in TruckData.iterrows():
                                truck = truck_row['TRUCK']
                                cap = truck_row['CAPACITY']
                                assigned = TripPlan[TripPlan[truck_col] == truck]
                                total_cyl = assigned[cyl_col].sum()
                                num_trips = assigned.shape[0] if assigned.shape[0] > 0 else 1
                                total_hr = assigned[hr_col].sum()
                                cyl_util = (total_cyl / cap * 100) / num_trips if cap > 0 else 0
                                hr_util = (total_hr / maxhr * 100)
                                if truck not in cyl_utilization:
                                    cyl_utilization[truck] = {}
                                    hr_utilization[truck] = {}
                                cyl_utilization[truck][day] = round(cyl_util, 1)
                                hr_utilization[truck][day] = round(hr_util, 1)

                        # Cyl Utilization
                        cyl_util_df = pd.DataFrame.from_dict(cyl_utilization, orient='index', columns=days)
                        cyl_util_df['Avg'] = cyl_util_df.replace('', np.nan).replace(0, np.nan).mean(axis=1, skipna=True).round(1)
                        cyl_util_df = cyl_util_df.sort_values(by='Avg', ascending=False)
                        # Add a Total row as average of non-blank and non-zero values in each column
                        total_row = {}
                        for col in cyl_util_df.columns:
                            if col != 'Avg':
                                vals = cyl_util_df[col].replace('', np.nan).replace(0, np.nan).dropna()
                                total_row[col] = round(vals.mean(), 1) if not vals.empty else np.nan
                            else:
                                vals = cyl_util_df['Avg'].replace('', np.nan).replace(0, np.nan).dropna()
                                total_row['Avg'] = round(vals.mean(), 1) if not vals.empty else np.nan
                        total_row = pd.DataFrame([total_row], index=['Total'])
                        cyl_util_df = pd.concat([cyl_util_df, total_row])
                        

                        # Hour Utilization
                        hr_util_df = pd.DataFrame.from_dict(hr_utilization, orient='index', columns=days)
                        hr_util_df['Avg'] = hr_util_df.replace('', np.nan).replace(0, np.nan).mean(axis=1, skipna=True).round(1)
                        hr_util_df = hr_util_df.sort_values(by='Avg', ascending=False)
                        # Add a Total row as average of non-blank and non-zero values in each column
                        total_row = {}
                        for col in hr_util_df.columns:
                            if col != 'Avg':
                                vals = hr_util_df[col].replace('', np.nan).replace(0, np.nan).dropna()
                                total_row[col] = round(vals.mean(), 1) if not vals.empty else np.nan
                            else:
                                vals = hr_util_df['Avg'].replace('', np.nan).replace(0, np.nan).dropna()
                                total_row['Avg'] = round(vals.mean(), 1) if not vals.empty else np.nan   
                        total_row = pd.DataFrame([total_row], index=['Total'])
                        hr_util_df = pd.concat([hr_util_df, total_row])
                        
                        def cyl_util_color(val):
                            if pd.isna(val):
                                return ''
                            if val >= 90:
                                color = '#ff4136'  
                            elif val >= 70:
                                color = '#ffeb3b'  
                            elif val > 50:
                                color = '#2ecc40'  
                            elif val == 0:
                                color = "#919291"                              
                            else:
                                color = "#9893DB"
                            return f'background-color: {color}'

                        col1,col2=st.columns([1,1])
                        col1.caption("Cylinder Utilization (%) per Truck")
                        col1.dataframe(
                            cyl_util_df.style
                                .format("{:.1f}")
                                .applymap(cyl_util_color, subset=days + ['Avg']),
                            hide_index=False, width='content'
                        )

                        col2.caption("Hour Utilization (%) per Truck")
                        col2.dataframe(
                            hr_util_df.style
                                .format("{:.1f}")
                                .applymap(cyl_util_color, subset=days + ['Avg']),
                            hide_index=False, width='content'
                        )

                with st.expander("Monthly Cost Summary", expanded=True):
                    #----------------------------------------------------------------
                    # Monthly Summary & Cost Calculation
                    #----------------------------------------------------------------
                    # Calculate Cyl2 for each day as Cyl * Cyl/Trip (to calculuate monthly cyl taking less than 1 trip in a week into account per truck)
                    for day in days: TripPlan[f'{day}Cyl_M'] = TripPlan[f'{day}Cyl'] * TripPlan[day] * 4
                    for day in days: TripPlan[f'{day}Km_M'] = TripPlan[f'{day}Km'] * TripPlan[day] * 4
                    # Calculate total DayCyl_M and DayKm_M for each truck and day
                    truck_day_totals = []
                    # Calculate total monthly cylinders and kilometers for each truck (summed over all days)
                    for _, truck_row in TruckData.iterrows():
                        truck = truck_row['TRUCK']
                        total_cyl = 0
                        total_km = 0
                        for day in days:
                            truck_col = f"{day}Truck"
                            cyl_col = f"{day}Cyl_M"
                            km_col = f"{day}Km_M"
                            total_cyl += TripPlan.loc[TripPlan[truck_col] == truck, cyl_col].sum()
                            total_km += TripPlan.loc[TripPlan[truck_col] == truck, km_col].sum()
                        truck_day_totals.append({
                            "Truck": truck,
                            "MonthCyl": total_cyl,
                            "MonthKm": total_km
                        })
                    truck_summary = pd.DataFrame(truck_day_totals)
                    # Add FIXFEE, CYLFEE, KMFEE from TruckData to truck_summary
                    if TruckData is not None and not TruckData.empty:
                        truck_summary = truck_summary.merge(
                            TruckData[['TRUCK', 'FIXFEE', 'CYLFEE', 'KMFEE']],
                            left_on='Truck',
                            right_on='TRUCK',
                            how='left'
                        ).drop(columns=['TRUCK'])
                    truck_summary['MONTHLY_COST']= (truck_summary['FIXFEE'] + (truck_summary['MonthCyl'] * truck_summary['CYLFEE']) + (truck_summary['MonthKm'] * truck_summary['KMFEE'])).round(0)
                    # Add a total row to truck_summary
                    total_row = {
                        "Truck": "Total",
                        "MonthCyl": truck_summary["MonthCyl"].sum(),
                        "MonthKm": truck_summary["MonthKm"].sum(),
                        "FIXFEE": truck_summary["FIXFEE"].sum(),
                        "CYLFEE": truck_summary["CYLFEE"].mean(),
                        "KMFEE": truck_summary["KMFEE"].mean(),
                        "MONTHLY_COST": truck_summary["MONTHLY_COST"].sum()
                    }
                    truck_summary = pd.concat([truck_summary, pd.DataFrame([total_row])], ignore_index=True)
                    truck_summary['Cost/Cyl']= (truck_summary['MONTHLY_COST'] / truck_summary['MonthCyl']).replace([np.inf, -np.inf], 0).round(2)
                    st.dataframe(truck_summary, width='content',height=400,hide_index=True)
        

    
