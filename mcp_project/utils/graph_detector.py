import re
import json
import ast
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import streamlit as st
from typing import Dict, Any, Optional, List, Tuple
import logging
import numpy as np
from datetime import datetime, timedelta


def detect_graph_patterns(text: str) -> List[Dict[str, Any]]:
    """
    Detect various graph patterns in LLM response text
    """
    graphs = []

    # Pattern 1: JSON-like graph data (most common)
    json_patterns = [
        r'```json\s*(\{.*?\})\s*```',
        r'```\s*(\{.*?\})\s*```',
        r'\{.*?"type":\s*"(bar|line|scatter|pie|histogram)".*?\}',
        r'\{.*?"x":\s*\{.*?".*?".*?\}.*?"y":\s*\{.*?".*?".*?\}.*?\}',  # Nested x,y objects
        r'\{.*?"historical":\s*\[.*?\].*?"forecast":\s*\[.*?\].*?\}'  # Historical/forecast arrays
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                graph_data = json.loads(match)
                if is_valid_graph_data(graph_data):
                    graphs.append({
                        'type': 'plotly',
                        'data': graph_data,
                        'source': 'json_pattern'
                    })
            except json.JSONDecodeError:
                try:
                    # Try parsing as Python dict
                    graph_data = ast.literal_eval(match)
                    if is_valid_graph_data(graph_data):
                        graphs.append({
                            'type': 'plotly',
                            'data': graph_data,
                            'source': 'python_dict'
                        })
                except:
                    continue

    # Pattern 2: CSV-like data with actual data points
    csv_pattern = r'```(?:csv)?\s*([\w\s,]+\n(?:\d+[\.\d]*\s*,\s*[-]?\d+[\.\d]*\n)+)```'
    csv_matches = re.findall(csv_pattern, text)
    for csv_data in csv_matches:
        try:
            # Check if it has actual numeric data, not just text
            lines = csv_data.strip().split('\n')
            if len(lines) > 2 and any(any(char.isdigit() for char in line) for line in lines[1:]):
                graphs.append({
                    'type': 'csv',
                    'data': csv_data,
                    'source': 'csv_pattern'
                })
        except:
            continue

    # Pattern 3: Simple data arrays with actual numbers
    array_pattern = r'x:\s*\[([\d\.,\s\-]+)\]\s*y:\s*\[([\d\.,\s\-]+)\]'
    array_matches = re.findall(array_pattern, text)
    for x_data, y_data in array_matches:
        try:
            # Check if we have actual numeric data
            x_values = [float(x.strip()) for x in x_data.split(',') if
                        x.strip().replace('-', '').replace('.', '').isdigit()]
            y_values = [float(y.strip()) for y in y_data.split(',') if
                        y.strip().replace('-', '').replace('.', '').isdigit()]

            if len(x_values) > 1 and len(y_values) > 1 and len(x_values) == len(y_values):
                graphs.append({
                    'type': 'simple_xy',
                    'data': {'x': x_values, 'y': y_values},
                    'source': 'array_pattern'
                })
        except:
            continue

    # Pattern 4: Forecast data with actual numbers (not just text description)
    forecast_pattern = r'forecast.*?data.*?\[([\d\.,\s\-]+)\].*?\[([\d\.,\s\-]+)\]'
    forecast_matches = re.findall(forecast_pattern, text, re.IGNORECASE | re.DOTALL)
    for forecast_x, forecast_y in forecast_matches:
        try:
            x_values = [float(x.strip()) for x in forecast_x.split(',') if
                        x.strip().replace('-', '').replace('.', '').isdigit()]
            y_values = [float(y.strip()) for y in forecast_y.split(',') if
                        y.strip().replace('-', '').replace('.', '').isdigit()]

            if len(x_values) > 1 and len(y_values) > 1:
                graphs.append({
                    'type': 'forecast_data',
                    'data': {'x': x_values, 'y': y_values, 'forecast_text': text},
                    'source': 'forecast_pattern'
                })
        except:
            continue

    return graphs


def is_valid_graph_data(data: Any) -> bool:
    """Check if the data contains actual graph data, not just metadata"""
    if not isinstance(data, dict):
        return False

    # Check for the new nested structure
    has_nested_data = (
            ('x' in data and isinstance(data['x'], dict) and
             any(isinstance(v, list) and len(v) > 1 for v in data['x'].values())) or
            ('y' in data and isinstance(data['y'], dict) and
             any(isinstance(v, list) and len(v) > 1 for v in data['y'].values())) or
            ('historical' in data and isinstance(data['historical'], list) and len(data['historical']) > 1) or
            ('forecast' in data and isinstance(data['forecast'], list) and len(data['forecast']) > 1)
    )

    # Check for traditional flat structure
    has_flat_data = (
            ('x' in data and isinstance(data['x'], list) and len(data['x']) > 1) or
            ('y' in data and isinstance(data['y'], list) and len(data['y']) > 1) or
            ('data' in data and isinstance(data['data'], list) and len(data['data']) > 0) or
            ('values' in data and isinstance(data['values'], list) and len(data['values']) > 1)
    )

    return has_nested_data or has_flat_data


def generate_forecast_from_text(forecast_text: str) -> Optional[Dict[str, Any]]:
    """
    Generate forecast graph data from text description
    when actual data arrays aren't provided
    """
    try:
        # Extract forecast points from text
        forecast_points = []
        date_pattern = r'(\d+)\s*(?:day|month).*?\$([\d\.]+)'
        matches = re.findall(date_pattern, forecast_text, re.IGNORECASE)

        for days, price in matches:
            forecast_points.append({
                'days': int(days),
                'price': float(price)
            })

        if not forecast_points:
            return None

        # Generate synthetic data for visualization
        base_date = datetime.now()
        dates = [base_date + timedelta(days=point['days']) for point in forecast_points]
        prices = [point['price'] for point in forecast_points]

        return {
            'type': 'line',
            'x': [date.strftime('%Y-%m-%d') for date in dates],
            'y': prices,
            'title': 'Price Forecast',
            'labels': {'x': 'Date', 'y': 'Price'}
        }

    except Exception as e:
        logging.error(f"Error generating forecast from text: {e}")
        return None


def create_nested_structure_chart(data: Dict[str, Any], graph_type: str) -> go.Figure:
    """
    Create a chart from the nested historical/forecast structure
    """
    fig = go.Figure()

    # Add historical data
    if 'x' in data and 'historical' in data['x'] and 'y' in data and 'historical' in data['y']:
        fig.add_trace(go.Scatter(
            x=data['x']['historical'],
            y=data['y']['historical'],
            mode='lines',
            name='Historical',
            line=dict(color='blue')
        ))

    # Add forecast data
    if 'x' in data and 'forecast' in data['x'] and 'y' in data and 'forecast' in data['y']:
        fig.add_trace(go.Scatter(
            x=data['x']['forecast'],
            y=data['y']['forecast'],
            mode='lines',
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))

    # Add confidence intervals if available
    if 'lower' in data and 'upper' in data and 'x' in data and 'forecast' in data['x']:
        fig.add_trace(go.Scatter(
            x=data['x']['forecast'] + data['x']['forecast'][::-1],  # Reverse for filled area
            y=data['upper'] + data['lower'][::-1],
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence'
        ))

    # Set title and labels
    title = data.get('title', 'Forecast Chart')
    fig.update_layout(
        title=title,
        xaxis_title=data.get('layout', {}).get('xaxis', {}).get('title', 'Date'),
        yaxis_title=data.get('layout', {}).get('yaxis', {}).get('title', 'Price'),
        legend=dict(orientation=data.get('layout', {}).get('legend', {}).get('orientation', 'h'))
    )

    return fig
def render_detected_graph(graph_info: Dict[str, Any], graph_index: int = 0) -> bool:
    """
    Render a detected graph based on its type and data
    """
    try:
        # Generate a unique key for this graph
        unique_key = f"graph_{graph_index}_{hash(str(graph_info))}"

        if graph_info['type'] == 'plotly':
            data = graph_info['data']
            graph_type = data.get('type', 'line')

            # Handle the new nested structure with historical/forecast data
            if 'x' in data and isinstance(data['x'], dict) and 'y' in data and isinstance(data['y'], dict):
                fig = create_nested_structure_chart(data, graph_type)
            else:
                # Handle traditional flat structure
                if graph_type == 'bar':
                    fig = px.bar(data, x=data.get('x'), y=data.get('y'),
                                 title=data.get('title', 'Bar Chart'))
                elif graph_type == 'line':
                    fig = px.line(data, x=data.get('x'), y=data.get('y'),
                                  title=data.get('title', 'Line Chart'))
                elif graph_type == 'scatter':
                    fig = px.scatter(data, x=data.get('x'), y=data.get('y'),
                                     title=data.get('title', 'Scatter Plot'))
                else:
                    fig = go.Figure(data=data.get('data', []),
                                    layout=data.get('layout', {}))

            # Apply custom layout if provided
            if 'layout' in data:
                fig.update_layout(**data['layout'])

            st.plotly_chart(fig, use_container_width=True, key=unique_key)
            return True

        elif graph_info['type'] == 'csv':
            df = pd.read_csv(pd.compat.StringIO(graph_info['data']))
            if len(df.columns) >= 2:
                fig = px.line(df, x=df.columns[0], y=df.columns[1],
                              title='Data Visualization')
                st.plotly_chart(fig, use_container_width=True, key=unique_key)
                return True

        elif graph_info['type'] == 'simple_xy':
            data = graph_info['data']
            df = pd.DataFrame({'x': data['x'], 'y': data['y']})
            fig = px.line(df, x='x', y='y', title='Data Trend')
            st.plotly_chart(fig, use_container_width=True, key=unique_key)
            return True

        elif graph_info['type'] == 'forecast_data':
            data = graph_info['data']
            df = pd.DataFrame({'x': data['x'], 'y': data['y']})
            fig = px.line(df, x='x', y='y', title='Forecast Data')
            st.plotly_chart(fig, use_container_width=True, key=unique_key)
            return True

    except Exception as e:
        logging.error(f"Error rendering graph: {e}")
        return False

    return False


def extract_and_render_graphs(text: str) -> bool:
    """
    Main function to extract and render all graphs from text
    Returns True if any graphs were rendered
    """
    graphs = detect_graph_patterns(text)
    rendered_count = 0

    # If no graphs found but text contains forecast description, generate one
    if not graphs and any(word in text.lower() for word in ['forecast', 'prediction', 'arima']):
        generated_forecast = generate_forecast_from_text(text)
        if generated_forecast:
            graphs.append({
                'type': 'plotly',
                'data': generated_forecast,
                'source': 'generated_from_text'
            })

    if graphs:
        st.markdown("---")
        st.markdown("### 📊 Interactive Visualizations")

        for i, graph_info in enumerate(graphs):
            if render_detected_graph(graph_info, i):  # Pass the index for unique key
                rendered_count += 1
                if i < len(graphs) - 1:
                    st.markdown("---")

    return rendered_count > 0