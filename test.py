# Dashboard
import streamlit as st

# Common:
import pandas as pd
import numpy as np
import statistics


# Visualization
import plotly.express as px
# import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

################################################################## Instuctions ##################################################################
#  - To run the visualization write the following command in the terminal: streamlit run test.py



################################################################## Dashboard Settings ##################################################################
st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 1400px;  /* Adjust this value as needed */
        padding-left: 2rem;     
        padding-right: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)



################################################################## Visualization Functions ##################################################################
def map_viz(df,color_feature):
    ######################################################################### Color #########################################################################
    color_features = {
        "Traffic Affect Severity": "Severity",
        "Hour": "Hour",
        "Day or Night": "Civil_Twilight",
    }

    color_feature = color_features[color_feature]

    if color_feature == 'Severity':
        color_discrete_map = {
            '1': '#28B463',
            '2': '#3498DB',
            '3': '#E74C3C',
            '4': '#17202A'
        }
        df['Severity'] = df['Severity'].astype(str)

    elif color_feature == "Hour":
        color_scale = ['#000000', '#00001a', '#000033', '#00004d', '#000066', '#990000', '#e62e00', '#ff8000',
                       '#ffb31a', '#66ccff', '#1ab2ff', '#0099ff']

        color_scale = color_scale + color_scale[::-1]

        color_scale = ["#000000","#00E1E5","#F40000","#000000"]
        color_discrete_map = None

    elif color_feature == "Civil_Twilight":
        color_discrete_map = {
            'Day': "#FFB200",
            'Night': "#000000"
        }


    if color_feature in ['Civil_Twilight', 'Severity']:
        fig = px.scatter_mapbox(
            df,
            lat='Start_Lat',
            lon='Start_Lng',
            color=color_feature,
            color_discrete_map=color_discrete_map,
            size_max=15,
            zoom=3,
            mapbox_style="carto-positron",
            title="Geospatial Visualization of Accidents",
            hover_data={'State': True, 'Country': True, 'City': True, 'Street': True},
            opacity=0.5
        )
    else:
        fig = px.scatter_mapbox(
            df,
            lat='Start_Lat',
            lon='Start_Lng',
            color=color_feature,
            color_continuous_scale=color_scale,
            size_max=15,
            zoom=3,
            mapbox_style="carto-positron",
            title="Geospatial Visualization of Accidents",
            hover_data={'State': True, 'Country': True, 'City': True, 'Street': True},
            opacity=1
        )


    # Update the layout of the figure
    fig.update_layout(
        width=1400,
        height=800,
        template='seaborn',
        font=dict(family='Arial', size=12, color='black'),  # Set font style
    )

    return fig

def feature_counts_plot(df, feature_name):
    # Mapping feature names to their respective columns in the DataFrame
    feature_name_mappings = {
        "Traffic Affect Severity": "Severity",
        "Hour": "Hour",
        "Day or Night": "Civil_Twilight",
    }

    # Retrieve the actual column name from the mappings
    feature_name = feature_name_mappings[feature_name]

    # Count occurrences of each value in the feature column
    feature_counts = df[feature_name].value_counts()

    # Create a DataFrame from the counts
    feature_counts_df = feature_counts.reset_index()
    feature_counts_df.columns = [feature_name, 'Count']

    # Sort values by count
    feature_counts_df = feature_counts_df.sort_values(by='Count', ascending=True)

    if feature_name == 'Severity':
        color_discrete_map = {
            '1': '#28B463',
            '2': '#3498DB',
            '3': '#E74C3C',
            '4': '#17202A'
        }
        feature_counts_df[feature_name] = feature_counts_df[feature_name].astype(str)  # Ensure string type for mapping
        fig = px.bar(
            feature_counts_df,
            x=feature_name,
            y='Count',
            orientation='v',
            title=f'Count of {feature_name} Values',
            labels={'Count': 'Count', feature_name: feature_name},
            color=feature_name,  # Color based on the feature column
            color_discrete_map=color_discrete_map if feature_name not in ['Hour'] else None,  # Apply color mappings
            width=400,  # Adjust width of bars (default is 100%)
        )





    elif feature_name == 'Hour':
        # Define a color scale based on the counts
        color_scale = px.colors.sequential.amp

        fig = px.bar(
            feature_counts_df,
            x=feature_name,
            y='Count',
            orientation='v',
            title=f'Count of {feature_name} Values',
            labels={'Count': 'Count', feature_name: feature_name},
            color='Count',  # Color based on the 'Count' column
            color_continuous_scale=color_scale,  # Assign the chosen color scale
            width=800,  # Adjust width of bars
        )



    elif feature_name == "Civil_Twilight":
        color_discrete_map = {
            'Day': "#FFB200",
            'Night': "#000000"
        }

        fig = px.bar(
            feature_counts_df,
            x=feature_name,
            y='Count',
            orientation='v',
            title=f'Count of {feature_name} Values',
            labels={'Count': 'Count', feature_name: feature_name},
            color=feature_name,  # Color based on the feature column
            color_discrete_map=color_discrete_map if feature_name not in ['Hour'] else None,  # Apply color mappings
            width=400,  # Adjust width of bars (default is 100%)
        )

    # Update layout for better styling
    fig.update_layout(
        width=800,
        height=770,
        template='plotly_white',
        xaxis_title=feature_name,
        yaxis_title='Count',
        showlegend=False,
        plot_bgcolor='#f2f2f2',  # Set plot background color
        paper_bgcolor='white',  # Set paper background color
        font=dict(family='Arial', size=12, color='black'),  # Set font style
    )

    return fig

def sign_plot(df, list_of_signs):
    ######################################################################### Prep #########################################################################

    ############### custom color sequence ###############
    severity_colors = {
        '1': '#D98880',
        '2': '#CD6155',
        '3': '#922B21',
        '4': '#641E16'
    }

    ############### initializing sub plots ###############
    fig = make_subplots(
        rows=2, cols=len(list_of_signs),
        shared_yaxes=False,
        vertical_spacing=0.15,  # Reduce vertical spacing
        subplot_titles=list_of_signs,
        specs=[[{"type": "bar"}] * len(list_of_signs), [{"type": "pie"}] * len(list_of_signs)],
        row_heights=[0.8, 0.2]  # Adjust the heights: 80% for bar plots, 20% for pie charts
    )

    ######################################################################### filters #########################################################################
    ############### filter by hour ###############
    morning_hours = [6, 7, 8, 9, 10, 11]
    afternoon_hours = [12, 13, 14, 15, 16]
    evening_hours = [17, 18, 19]
    night_hours = [20, 21, 22, 23, 0, 1, 2, 3, 4, 5]

    filter_to_all = True
    filter_to_morning = False
    filter_to_afternoon = False
    filter_to_evening = False
    filter_to_night = False

    if not filter_to_all:
        if filter_to_morning:
            df = df[df["Hour"].isin(morning_hours)]
        elif filter_to_afternoon:
            df = df[df["Hour"].isin(afternoon_hours)]
        elif filter_to_evening:
            df = df[df["Hour"].isin(evening_hours)]
        elif filter_to_night:
            df = df[df["Hour"].isin(night_hours)]

    # Generate bar plots for each sign
    for i, sign in enumerate(list_of_signs, start=1):
        # Filter the DataFrame for the current sign
        filtered_df = df[df[sign] == True]

        # Calculate the total count for the current sign
        total_count = len(filtered_df)

        # Get the value counts of 'Severity' for the current sign and normalize
        value_counts = filtered_df['Severity'].value_counts(normalize=True).sort_index().reset_index()
        value_counts.columns = ['Severity', 'percentage']

        # Convert Severity to string for better categorical plotting
        value_counts['Severity'] = value_counts['Severity'].astype(str)

        # Add bar trace to the subplot
        for severity in value_counts['Severity']:
            fig.add_trace(
                go.Bar(
                    x=[severity],
                    y=[value_counts.loc[value_counts['Severity'] == severity, 'percentage'].values[0] * 100],
                    name=severity,
                    marker_color=severity_colors[severity],
                    legendgroup=severity,  # Group traces by severity
                    showlegend=(i == 1)  # Show legend only in the first subplot
                ),
                row=1, col=i
            )

        # Calculate the percentage of samples with this sign as True
        sign_true_percentage = total_count / len(df) * 100

        # Add pie chart trace to the subplot
        fig.add_trace(
            go.Pie(
                labels=["True", "False"],
                values=[sign_true_percentage, 100 - sign_true_percentage],
                marker=dict(colors=['#1f77b4', '#d3d3d3']),
                legendgroup='Frequency',  # Group all pie charts under the same legend group
                showlegend=(i == 1),  # Show legend only in the first subplot
                textinfo='none',
                hoverinfo='label+percent',
                title=dict(text='Frequency')
            ),
            row=2, col=i
        )

    # Update layout
    fig.update_layout(
        title_text="Distribution of Severity Values for Each Sign with Frequency of Signs",
        barmode='group',
        xaxis_title='Severity',
        yaxis_title='Percentage',
        yaxis_ticksuffix='%',
        height=600
    )

    # Update the layout to add the legend
    fig.update_layout(
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.15,  # Adjust y position of the legend
            xanchor='center',
            x=0.5
        ),
    )

    return fig

def categorical_weather_plot(df,x_feature):
    # Extract the date from 'Start_Time'
    df['Date'] = df['Start_Time'].dt.date

    # Calculate the number of unique days each weather condition appeared
    unique_days_per_weather = df.groupby(x_feature)['Date'].nunique().reset_index(name='Unique_Days')

    accident_counts = df.groupby(x_feature).size().reset_index(name='Accidents')

    # Merge the unique days with accident counts
    merged_df = pd.merge(accident_counts, unique_days_per_weather, on=x_feature)

    # Normalize the accident counts
    merged_df['Accident_Rate'] = merged_df['Accidents'] / merged_df['Unique_Days']

    # Threshold Calculation
    rate_threshold = statistics.median(merged_df['Accident_Rate'].tolist())

    merged_df = merged_df[merged_df['Accident_Rate'] >= rate_threshold]

    # Sort the dataframe by the normalized accident rate in descending order
    merged_df = merged_df.sort_values(by='Accident_Rate', ascending=False)

    # Create the bar plot
    fig = px.bar(merged_df, x=x_feature, y='Accident_Rate',
                 title='Normalized Accident Rate by Weather Condition',
                 labels={'Weather_Condition': 'Weather Condition', 'Accident_Rate': 'Normalized Accident Rate'},
                 color='Weather_Condition' if x_feature=="Weather_Condition" else None,  # Adding color to make it visually appealing
                 template='plotly')  # You can use different templates like 'plotly_dark', 'ggplot2', etc.

    # Customize the layout for better visual appeal
    fig.update_layout(
        xaxis_title='Weather Condition' if x_feature == 'Weather_Condition' else 'Wind Direction',
        yaxis_title='Normalized Accident Rate',
        title={'x': 0.5, 'xanchor': 'center'},
        bargap=0.2,  # Space between bars
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
    )

    return fig

def continuous_weather_plot(df, continuous_feature):
    # Create a histogram to show the distribution of temperature values
    fig = px.histogram(
        df,
        x=continuous_feature,
        nbins=100,
        title=f'Distribution of {continuous_feature}',
        color_discrete_sequence=['#636EFA'],  # Custom color
    )

    # Update layout for better styling
    fig.update_layout(
        title={
            'text': f'Distribution of {continuous_feature}',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title=f'{continuous_feature.capitalize()}',
        yaxis_title='Traffic Accident Count',
        font=dict(
            family="Arial, sans-serif",
            size=14,
            color="#7f7f7f"
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=40, r=40, t=80, b=40)
    )

    # Update x-axis for better styling
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGray',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='LightGray'
    )

    # Update y-axis for better styling
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGray',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='LightGray'
    )

    return fig

def holidays_plot(df_grouped):
    # Creating the combined accident rate DataFrame
    df_combined = df_grouped.groupby(['Year', 'Holiday'])['Average_Daily_Accident_Count'].sum().reset_index()

    # Creating separate dataframes for Day and Night
    df_day = df_grouped[df_grouped['Sunrise_Sunset'] == 'Day'].groupby(['Year', 'Holiday'])[
        'Average_Daily_Accident_Count'].sum().reset_index()
    df_night = df_grouped[df_grouped['Sunrise_Sunset'] == 'Night'].groupby(['Year', 'Holiday'])[
        'Average_Daily_Accident_Count'].sum().reset_index()

    # Creating subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Average Daily Accident Count per Year & Holiday",
            "During the Day",
            "During the Night"
        ),
        specs=[[{"type": "bar", "colspan": 2}, None], [{"type": "bar"}, {"type": "bar"}]],
        shared_xaxes=True
    )

    # Define specific colors for each holiday
    holiday_colors = {
        "New Year's": "#aa6f73",
        'Independence Day': "#a39193",
        'Thanksgiving': "#eea990",
        'Christmas': "#f6e0b5",
        'Non Holiday': "#66545e"
    }

    # Define specific order of holidays
    holiday_order = ["New Year's", 'Independence Day', 'Thanksgiving', 'Christmas', 'Non Holiday']

    # Adding the combined accident rates bar plot with specific colors and order
    for holiday in holiday_order:
        if holiday in df_combined['Holiday'].unique():
            holiday_data = df_combined[df_combined['Holiday'] == holiday]
            fig.add_trace(
                go.Bar(name=holiday, x=holiday_data['Year'], y=holiday_data['Average_Daily_Accident_Count'],
                       legendgroup=holiday, marker_color=holiday_colors[holiday]),
                row=1, col=1
            )

    # Adding the day accident rates bar plot with specific colors and order
    for holiday in holiday_order:
        if holiday in df_day['Holiday'].unique():
            holiday_data = df_day[df_day['Holiday'] == holiday]
            fig.add_trace(
                go.Bar(name=holiday, x=holiday_data['Year'], y=holiday_data['Average_Daily_Accident_Count'],
                       legendgroup=holiday, showlegend=False, marker_color=holiday_colors[holiday]),
                row=2, col=1
            )

    # Adding the night accident rates bar plot with specific colors and order
    for holiday in holiday_order:
        if holiday in df_night['Holiday'].unique():
            holiday_data = df_night[df_night['Holiday'] == holiday]
            fig.add_trace(
                go.Bar(name=holiday, x=holiday_data['Year'], y=holiday_data['Average_Daily_Accident_Count'],
                       legendgroup=holiday, showlegend=False, marker_color=holiday_colors[holiday]),
                row=2, col=2
            )

    # Updating layout
    fig.update_layout(
        height=800,
        title_text="",
        barmode='group',
        template='seaborn'
    )

    return fig

def time_and_corona_trend_plot(df,states,lock_dates):
    def create_dataframe_for_plot(df,states,lock_dates):
        # State Dataset
        df.rename(columns={"State": "State_Abb"}, inplace=True)
        df = pd.merge(df, states, left_on="State_Abb", right_on="Abbreviation")
        df.drop(columns=["Abbreviation"], inplace=True)
        df.rename(columns={"State(Territory)": "State"}, inplace=True)
        # This df is not really needed, it's just for checking state-abb pairs.
        unique_pairs = df[['State', 'State_Abb']].drop_duplicates()

        # Corona Lockdown Dataset
        # Handle special case for Kansas City
        lock_dates['City'] = lock_dates['State'].apply(
            lambda x: 'Kansas City' if x == 'Kansas City in Kansas' else None)
        lock_dates['State'] = lock_dates['State'].replace('Kansas City in Kansas', 'Kansas')
        lock_dates['City'] = lock_dates['City'].replace('None', None)
        # Separate entries with City as None and with specific cities
        lock_dates_no_city = lock_dates[lock_dates['City'].isna()]
        lock_dates_with_city = lock_dates[~lock_dates['City'].isna()]
        # Merge the main DataFrame with the lock_dates where City is None
        merged_no_city = pd.merge(
            df,
            lock_dates_no_city[['State', 'Lockdown_Start_Date', 'Lockdown_End_Date', 'Lockdown_Length(days)']],
            on='State',
            how='left'
        )
        # Merge the result with the lock_dates where City is specified
        lockdown_df = pd.merge(
            merged_no_city,
            lock_dates_with_city[
                ['State', 'City', 'Lockdown_Start_Date', 'Lockdown_End_Date', 'Lockdown_Length(days)']],
            left_on=['State', 'City'],
            right_on=['State', 'City'],
            how='left',
            suffixes=('', '_city')
        )
        # Fill the lockdown dates from the city-specific merge where available
        lockdown_df['Lockdown_Start_Date'] = lockdown_df['Lockdown_Start_Date_city'].combine_first(
            lockdown_df['Lockdown_Start_Date'])
        lockdown_df['Lockdown_End_Date'] = lockdown_df['Lockdown_End_Date_city'].combine_first(
            lockdown_df['Lockdown_End_Date'])
        lockdown_df['Lockdown_Length(days)'] = lockdown_df['Lockdown_Length(days)_city'].combine_first(
            lockdown_df['Lockdown_Length(days)'])
        # Drop the intermediate columns
        lockdown_df.drop(columns=['Lockdown_Start_Date_city', 'Lockdown_End_Date_city', 'Lockdown_Length(days)_city'],
                         inplace=True)

        return lockdown_df

    lockdown_df = create_dataframe_for_plot(df,states,lock_dates)

    # Step 1: Convert the 'Start_Time' and lockdown date columns to datetime format
    lockdown_df['Start_Time'] = pd.to_datetime(lockdown_df['Start_Time'], errors='coerce')
    lockdown_df['Lockdown_Start_Date'] = pd.to_datetime(lockdown_df['Lockdown_Start_Date'], errors='coerce')
    lockdown_df['Lockdown_End_Date'] = pd.to_datetime(lockdown_df['Lockdown_End_Date'], errors='coerce')

    # Drop rows with NaT in 'Start_Time' after conversion
    lockdown_df = lockdown_df.dropna(subset=['Start_Time'])

    # Step 2: Create a new column for the month/year combination
    lockdown_df['Month_Year'] = lockdown_df['Start_Time'].dt.to_period('M')

    # Step 3: Create a new column to group severity levels
    def group_severity(severity):
        if severity in [1, 2]:
            return 'Severity 1-2'
        elif severity in [3, 4]:
            return 'Severity 3-4'

    lockdown_df['Severity_Group'] = lockdown_df['Severity'].apply(group_severity)

    # Filter the dataframe for the specified states and city
    states_and_city = ['California', 'Connecticut', 'Illinois', 'Kansas City', 'Massachusetts', 'Michigan', 'New York',
                       'Oregon', 'Wisconsin']
    lockdown_df_filtered = lockdown_df[
        (lockdown_df['State'].isin(states_and_city)) | (lockdown_df['City'] == 'Kansas City')]

    # Step 4: Aggregate the data by location, month/year, and severity group to get the count of accidents
    lockdown_df_filtered['Location'] = lockdown_df_filtered.apply(
        lambda x: x['City'] if x['City'] == 'Kansas City' else x['State'], axis=1)
    accidents_per_month_year_severity = lockdown_df_filtered.groupby(
        ['Location', 'Month_Year', 'Severity_Group']).size().reset_index(name='Accident_Count')

    # Convert Month_Year back to datetime for proper sorting and plotting
    accidents_per_month_year_severity['Month_Year'] = accidents_per_month_year_severity['Month_Year'].dt.to_timestamp()

    # Calculate the total number of accidents
    total_accidents = accidents_per_month_year_severity.groupby(['Location', 'Month_Year'])[
        'Accident_Count'].sum().reset_index(name='Total_Accidents')

    # Step 5: Create the initial Plotly line plot for all states and city
    fig = make_subplots()

    locations = accidents_per_month_year_severity['Location'].unique()
    severity_groups = accidents_per_month_year_severity['Severity_Group'].unique()

    # Create a line plot for each state and severity group, and add to the figure
    trace_visibility = {}
    for location in locations:
        trace_visibility[location] = []
        for severity_group in severity_groups:
            location_severity_data = accidents_per_month_year_severity[
                (accidents_per_month_year_severity['Location'] == location) &
                (accidents_per_month_year_severity['Severity_Group'] == severity_group)]
            trace = go.Scatter(x=location_severity_data['Month_Year'], y=location_severity_data['Accident_Count'],
                               mode='lines+markers', name=f'{location} - {severity_group}')
            fig.add_trace(trace)
            trace_visibility[location].append(trace)

        # Add the total accidents trace
        location_total_data = total_accidents[total_accidents['Location'] == location]
        total_trace = go.Scatter(x=location_total_data['Month_Year'], y=location_total_data['Total_Accidents'],
                                 mode='lines', name=f'{location} - Total Accidents',
                                 line=dict(color='black', width=2, dash='dash'))
        fig.add_trace(total_trace)
        trace_visibility[location].append(total_trace)

    # Create the dropdown buttons with lockdown lines
    dropdown_buttons = []
    for location in locations:
        lockdown_start = lockdown_df_filtered[lockdown_df_filtered['Location'] == location]['Lockdown_Start_Date'].iloc[
            0]
        lockdown_end = lockdown_df_filtered[lockdown_df_filtered['Location'] == location]['Lockdown_End_Date'].iloc[0]

        # Calculate X-axis range
        x_start = (lockdown_start - pd.DateOffset(months=6)).strftime('%Y-%m-%d')
        x_end = (lockdown_end + pd.DateOffset(months=6)).strftime('%Y-%m-%d')

        lockdown_lines = [
            {'type': 'line', 'x0': lockdown_start, 'x1': lockdown_start, 'y0': 0, 'y1': 1, 'xref': 'x', 'yref': 'paper',
             'line': {'color': 'black', 'width': 2, 'dash': 'dot'}, 'name': 'Lockdown Start'},
            {'type': 'line', 'x0': lockdown_end, 'x1': lockdown_end, 'y0': 0, 'y1': 1, 'xref': 'x', 'yref': 'paper',
             'line': {'color': 'green', 'width': 2, 'dash': 'dot'}, 'name': 'Lockdown End'}
        ]

        visibility = [False] * len(fig.data)
        for i, trace in enumerate(trace_visibility[location]):
            visibility[fig.data.index(trace)] = True

        dropdown_buttons.append({
            'label': location,
            'method': 'update',
            'args': [
                {'visible': visibility},
                {'title': f'Number of Accidents per Month/Year in {location}',
                 'shapes': lockdown_lines,
                 'xaxis': {'range': [x_start, x_end]}}
            ]
        })

    # Add a button for showing all locations without lockdown lines and default x-axis range
    default_x_start = (lockdown_df_filtered['Lockdown_Start_Date'].min() - pd.DateOffset(months=6)).strftime('%Y-%m-%d')
    default_x_end = (lockdown_df_filtered['Lockdown_End_Date'].max() + pd.DateOffset(months=6)).strftime('%Y-%m-%d')

    dropdown_buttons.append({'label': 'All Locations', 'method': 'update', 'args': [
        {'visible': [True] * len(fig.data)},
        {'title': 'Number of Accidents per Month/Year for All Locations',
         'shapes': [],
         'xaxis': {'range': [default_x_start, default_x_end]}}
    ]})

    # Update layout with dropdown menu and add custom legend items
    fig.update_layout(
        updatemenus=[{
            'buttons': dropdown_buttons,
            'direction': 'down',
            'showactive': True,
        }],
        xaxis_title='Month/Year',
        yaxis_title='Number of Accidents',
        title='Number of Accidents per Month/Year for All Locations',
        shapes=[
            # Add the lockdown lines here for the legend
            {'type': 'line', 'x0': pd.Timestamp('2020-03-01'), 'x1': pd.Timestamp('2020-03-01'), 'y0': 0, 'y1': 1,
             'xref': 'x', 'yref': 'paper',
             'line': {'color': 'black', 'width': 2, 'dash': 'dot'}, 'name': 'Lockdown Start'},
            {'type': 'line', 'x0': pd.Timestamp('2020-06-01'), 'x1': pd.Timestamp('2020-06-01'), 'y0': 0, 'y1': 1,
             'xref': 'x', 'yref': 'paper',
             'line': {'color': 'green', 'width': 2, 'dash': 'dot'}, 'name': 'Lockdown End'}
        ],
        annotations=[
            # Add annotations for the lockdown lines
            go.layout.Annotation(
                xref='paper', yref='paper', x=1.02, y=0.95, showarrow=False,
                text='Lockdown Start', font=dict(color='black'), align='left'
            ),
            go.layout.Annotation(
                xref='paper', yref='paper', x=1.02, y=0.90, showarrow=False,
                text='Lockdown End', font=dict(color='green'), align='left'
            ),
        ],
        legend=dict(
            yanchor="top",
            y=0.85,
            xanchor="left",
            x=1.0,
            title="",
            traceorder="grouped",
            font=dict(size=12),
            bgcolor="White"
        )
    )

    return fig


################################################################## Read Data ##################################################################

############## read data into dataframe ##############
data = pd.read_csv("data/US_Accidents_March23_sampled_500k.csv",low_memory=False) # Use Sample Data

############## add data & time ##############
# Remove milliseconds from datetime strings (if present)
data['Start_Time'] = data['Start_Time'].apply(lambda x: x.split('.')[0] if '.' in x else x)
data['End_Time'] = data['End_Time'].apply(lambda x: x.split('.')[0] if '.' in x else x)
# Convert the datetime columns to datetime object
data['Start_Time'] = pd.to_datetime(data['Start_Time'], format='%Y-%m-%d %H:%M:%S')
data['End_Time'] = pd.to_datetime(data['End_Time'], format='%Y-%m-%d %H:%M:%S')
# Create Date & Time Features
data['Year'] = data['Start_Time'].dt.year
data['Month'] = data['Start_Time'].dt.month
data['Day'] = data['Start_Time'].dt.day
data['Hour'] = data['Start_Time'].dt.hour

############## Add Holiday Data ##############
def defin_holidy(row):
    date = row['Date']

    # New Years
    if (date.month == 12 and date.day == 31) or (date.month == 1 and date.day == 1):
        return "New Year's"

    # Independence Day
    elif date.month == 7 and date.day == 4:
        return "Independence Day"

    # Thanksgiving
    elif str(date) in ['2017-11-23', '2017-11-24', '2018-11-22', '2018-11-23', '2019-11-28', '2019-11-29', '2020-11-26',
                       '2020-11-27', '2022-11-24', '2022-11-25']:
        return "Thanksgiving"

    # Christmas
    elif date.month == 12 and date.day == 25:
        return "Christmas"

    else:
        return "Non Holiday"
# Copy relevenet columns
df = data[["Start_Time","Year","Day","Sunrise_Sunset"]].copy()
# Extract the date part from 'Start_Time'
df['Date'] = df['Start_Time'].dt.date
# Create holidy column
df['Holiday'] = df.apply(defin_holidy, axis=1)
# Grouping by Year, Holiday, Sunrise_Sunset and calculating daily accident count and number of days
df_grouped = df.groupby(['Year', 'Holiday', 'Sunrise_Sunset']).agg(
    Daily_Accident_Count=('Date', 'size'),
    Number_of_Days=('Date', 'nunique')
).reset_index()
# Calculating average daily accident count
df_grouped['Average_Daily_Accident_Count'] = df_grouped['Daily_Accident_Count'] / df_grouped['Number_of_Days']
# Remove partly recorder years
df_grouped= df_grouped[(df_grouped['Year'] != 2023) & (df_grouped['Year'] != 2016)]
# Rename
holiday_average_daily_accident_count_df = df_grouped

############## Read Additional data ##############
states = pd.read_csv('data/state_abbreviations.csv')
lock_dates = pd.read_csv('data/US_Lockdown_Dates.csv')




################################################################## Dashboard Stucture ##################################################################

############## Side Bar ##############
st.sidebar.header("Settings")

# Date Ranges
start_date = st.sidebar.date_input("Start date", pd.to_datetime("01/01/2017", format='%d/%m/%Y'))
end_date = st.sidebar.date_input("End date", pd.to_datetime("31/12/2022", format='%d/%m/%Y'))

# Visualization Selector
visualizations = st.sidebar.multiselect(
    "Select visualizations to display",
    ["US Map & Accident Scatter", "Signs & Accidents", "Weather & Traffic Accidents", "Holidays & Traffic Accidents","Over Time Trends Including Coronavirus Lockdowns"],
    ["US Map & Accident Scatter", "Signs & Accidents", "Weather & Traffic Accidents", "Holidays & Traffic Accidents","Over Time Trends Including Coronavirus Lockdowns"] # defualt
)


# Filtered data based on selected date range
filtered_data = data[(data['Start_Time'] >= pd.to_datetime(start_date, format='%d/%m/%Y')) &
                     (data['Start_Time'] <= pd.to_datetime(end_date, format='%d/%m/%Y'))]


############## Dashboard ##############
st.title('Visualization Project - US Traffic Accidents')

st.header("Intro",divider='gray')
st.markdown("This dashboard is based on a USA traffic accidents dataset and includes several visualizations. Each plot is designed to answer specific questions and may allow to uncover additional insights and trends related to the subject.")
st.markdown("Note that the dataset used for this USA traffic accidents dashboard is a sample. The original dataset is too large to be presented effectively on this platform, which affects performance.")
st.markdown("We recommend using data from the years 2017 to 2022 inclusive, as data for the years 2016 and 2023 is partially recorded.")
st.markdown("Use the sidebar on the left to:\n"
            "1. Select the range of data (affects all except the holidays plot and corona plot due to constraints)\n"
            "2. Select visualizations to display\n"
            "3. Hide the sidebar")

st.markdown("The dynamic capabilities include:\n"
            "1. Clicking and double clicking on elements in the legend to hide or show elements.\n"
            "2. Clicking and draging on the plot to select specific regions to focus on in the plot\n"
            "3. Use mouse in the map plot to move and zoom in and out\n"
            "4. Double clicking on a plot to resting it to initial state.",unsafe_allow_html=True)

st.markdown("\n\n !!!!!!!!!!! If the dashboard is in dark mode, change it to light mode by pressing the 3 dots on the top right and going into settings !!!!!!!!!!! \n\n",unsafe_allow_html=True)

st.header("Dashboard",divider='gray')



if "US Map & Accident Scatter" in visualizations:
    st.subheader("US Accident Scatter Map")

    st.markdown("Question: Can we use the coordinate data to answer location based questions ?")

    st.markdown("In this question we chose 2 sub questions that require coordinates of traffic accidents to answer:\n"
                "1. Are there roads that their traffic is heavily affected by accidents ?\n"
                "2. Are there locations that are prone to accidents during the night or day ?")

    # Add dropdown for color feature selection
    color_feature = st.selectbox(
        "Select Feature",
        ["Traffic Affect Severity", "Day or Night"]
    )

    # Use columns to divide the layout
    col1, col2 = st.columns([0.7, 0.3])

    # Left column (col1)
    with col1:
        st.header('US Map - Accident Scatter')
        fig1 = map_viz(filtered_data, color_feature)
        st.plotly_chart(fig1)

    # Right column (col2)
    with col2:
        st.header('Value Frequency')
        fig2 = feature_counts_plot(filtered_data, color_feature)
        st.plotly_chart(fig2)

    st.markdown('---')


if "Signs & Accidents" in visualizations:
    st.subheader("Accidents Near Road Signs Bar Plot")

    st.markdown("Question: Is there an increased rate of traffic accidents near specific signs ?")

    st.markdown("The following plot contains 2 mains aspects.\n"
                "The first is the bar plots for each sign. Each sign hast a bar plot showing the distribution of accident severity of affect on road traffic. Note that the y axis is percentages and they sum up to 100.\n"
                "The second is the pi charts under each sign bar plot. Each plot shows the persentage of accidents from the intire dataset which happend near that road sign\n", unsafe_allow_html=True)

    list_of_signs = st.multiselect(
        "Select signs to display",
        ['Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop',
         'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop'],
        ['Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Station', 'Stop',
         'Traffic_Calming', 'Traffic_Signal']
    )

    fig = sign_plot(filtered_data,list_of_signs)
    st.plotly_chart(fig)

    st.markdown('---')


if "Weather & Traffic Accidents" in visualizations:

    st.subheader("Accidents During Weather Condition Bar Plot")

    st.markdown("Question: Are the weather conditions that have increased accident rates in them ?")

    st.markdown(
        "The following plot shows the normalized traffic accident rates during each of the weather conditions.\nNote that only weather conditions that have a greater rate then the midean are shown,",unsafe_allow_html=True)

    fig = categorical_weather_plot(data,"Weather_Condition")
    st.plotly_chart(fig)

    st.markdown('---')


if "Holidays & Traffic Accidents" in visualizations:

    st.subheader("Average Daily Accident Counts during holidays Over The Years")

    st.markdown("Question: Are there differences in average daily accident counts or trends between holiday and non-holiday periods, as well as between day and night times?")

    st.markdown(
        "The following plot explores the average daily accident counts during different holidays and non holidays accross different years.\n",
        unsafe_allow_html=True)

    fig = holidays_plot(holiday_average_daily_accident_count_df)
    st.plotly_chart(fig)

    st.markdown('---')


if "Over Time Trends Including Coronavirus Lockdowns" in visualizations:
    st.subheader("Traffic Accident Trends Over Time With Corona Lockdowns")

    st.markdown("Question: Is there a trend in the number of traffic accidents over the years, and how have coronavirus lockdowns impacted this trend?")

    st.markdown("The following plot depicts the number of accidents over the years, highlighting visible lockdown start and end dates. "
                "Due to different lockdown periods in each state, viewing the plot for all states may not provide significant insights. It is recommended to select a specific state from the dropdown on the left to view its plot.",unsafe_allow_html=True)
    st.markdown("The plot is zoomed in from the start and can be zoomed out by double clicking.",unsafe_allow_html=True)

    fig = time_and_corona_trend_plot(data,states,lock_dates)
    st.plotly_chart(fig)

    st.markdown('---')