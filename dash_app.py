import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression

# تحديد المسار الحالي للملف تلقائياً
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(BASE_DIR, 'customer_churn.csv')
model_path = os.path.join(BASE_DIR, 'churn_model.pkl')

# 1. Load dataset
try:
    df = pd.read_csv(data_path)
except Exception as e:
    df = pd.DataFrame()
    print(f"Error loading dataset: {e}")

# 2. Smart Model Loading with Auto-Fallback (Trains on the fly if file is incompatible)
model = None
try:
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("✅ Successfully loaded model from file!")
except Exception as e:
    print(f"⚠️ Could not load pickle file ({e}). Training fallback model on the fly...")

# If model failed to load or file is incompatible, train a quick Logistic Regression model right now!
if model is None and not df.empty:
    try:
        X = df[['Age', 'Total_Purchase', 'Account_Manager', 'Years', 'Num_Sites']]
        y = df['Churn']
        model = LogisticRegression()
        model.fit(X, y)
        print("✅ Fallback Logistic Regression model trained successfully on dataset!")
    except Exception as ex:
        print(f"Error training fallback model: {ex}")

# Preprocessing for metrics safety
if not df.empty and 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Initialize Dash application
app = dash.Dash(__name__)
server = app.server  # Required for cloud deployment (Gunicorn)
app.title = "Advanced Customer Churn Intelligence System"

# Calculate advanced metrics
total_customers = len(df) if not df.empty else 0
churn_count = int(df['Churn'].sum()) if not df.empty and 'Churn' in df.columns else 0
churn_rate = (churn_count / total_customers) * 100 if total_customers > 0 else 0
avg_purchase = df['Total_Purchase'].mean() if not df.empty and 'Total_Purchase' in df.columns else 0

# App Layout with high-end executive design
app.layout = html.Div(style={
    'backgroundColor': '#0b0f19', 
    'color': '#f3f4f6', 
    'padding': '40px', 
    'fontFamily': 'Inter, system-ui, sans-serif'
}, children=[
    
    # Header Section
    html.Div([
        html.H1("⚡ Advanced Customer Churn Intelligence & Analytics", style={'color': '#00ff88', 'marginBottom': '10px', 'fontWeight': '700'}),
        html.P("Executive-level predictive dashboard engineered with advanced analytics and real-time inference.", style={'color': '#9ca3af', 'fontSize': '16px'})
    ], style={'textAlign': 'center', 'marginBottom': '35px'}),
    
    # KPI Cards Section
    html.Div([
        html.Div([
            html.H4("Total Customers", style={'color': '#9ca3af', 'fontSize': '14px', 'margin': '0 0 8px 0'}),
            html.H2(f"{total_customers:,}", style={'color': '#00ff88', 'margin': '0', 'fontWeight': '700'})
        ], style={'backgroundColor': '#111827', 'padding': '24px', 'borderRadius': '12px', 'width': '22%', 'display': 'inline-block', 'textAlign': 'center', 'border': '1px solid #1f2937', 'boxShadow': '0 8px 20px rgba(0,255,136,0.15)'}),
        
        html.Div([
            html.H4("Churned Customers", style={'color': '#9ca3af', 'fontSize': '14px', 'margin': '0 0 8px 0'}),
            html.H2(f"{churn_count:,}", style={'color': '#ff4b4b', 'margin': '0', 'fontWeight': '700'})
        ], style={'backgroundColor': '#111827', 'padding': '24px', 'borderRadius': '12px', 'width': '22%', 'display': 'inline-block', 'textAlign': 'center', 'marginLeft': '3%', 'border': '1px solid #1f2937', 'boxShadow': '0 8px 20px rgba(255,75,75,0.15)'}),
        
        html.Div([
            html.H4("Churn Rate", style={'color': '#9ca3af', 'fontSize': '14px', 'margin': '0 0 8px 0'}),
            html.H2(f"{churn_rate:.2f}%", style={'color': '#fbbf24', 'margin': '0', 'fontWeight': '700'})
        ], style={'backgroundColor': '#111827', 'padding': '24px', 'borderRadius': '12px', 'width': '22%', 'display': 'inline-block', 'textAlign': 'center', 'marginLeft': '3%', 'border': '1px solid #1f2937', 'boxShadow': '0 8px 20px rgba(251,191,36,0.15)'}),
        
        html.Div([
            html.H4("Avg Total Purchase", style={'color': '#9ca3af', 'fontSize': '14px', 'margin': '0 0 8px 0'}),
            html.H2(f"${avg_purchase:,.2f}", style={'color': '#3b82f6', 'margin': '0', 'fontWeight': '700'})
        ], style={'backgroundColor': '#111827', 'padding': '24px', 'borderRadius': '12px', 'width': '22%', 'display': 'inline-block', 'textAlign': 'center', 'marginLeft': '3%', 'border': '1px solid #1f2937', 'boxShadow': '0 8px 20px rgba(59,130,246,0.15)'})
    ], style={'textAlign': 'center', 'marginBottom': '40px'}),
    
    # Charts Section 1
    html.Div([
        html.Div([
            dcc.Graph(
                figure=px.histogram(df, x='Account_Manager', color='Churn', barmode='group',
                                     title="📊 Churn Distribution by Account Manager",
                                     template="plotly_dark",
                                     color_discrete_map={0: '#00ff88', 1: '#ff4b4b'}).update_layout(
                    plot_bgcolor='#111827', paper_bgcolor='#111827', font_color='#f3f4f6', transition_duration=500
                ) if not df.empty else {}
            )
        ], style={'width': '48%', 'display': 'inline-block', 'backgroundColor': '#111827', 'padding': '20px', 'borderRadius': '12px', 'boxShadow': '0 10px 25px rgba(0,0,0,0.5)', 'border': '1px solid #1f2937'}),
        
        html.Div([
            dcc.Graph(
                figure=px.pie(df, names='Location', values='Total_Purchase' if not df.empty and 'Total_Purchase' in df.columns else None, 
                              title="🌐 Total Purchases Breakdown by Location",
                              template="plotly_dark", hole=0.4, color_discrete_sequence=['#00ff88', '#3b82f6', '#ff4b4b', '#fbbf24']).update_layout(
                    plot_bgcolor='#111827', paper_bgcolor='#111827', font_color='#f3f4f6', transition_duration=500
                ) if not df.empty else {}
            )
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right', 'backgroundColor': '#111827', 'padding': '20px', 'borderRadius': '12px', 'boxShadow': '0 10px 25px rgba(0,0,0,0.5)', 'border': '1px solid #1f2937'})
    ], style={'marginBottom': '40px'}),
    
    # Charts Section 2 (Scatter Plot)
    html.Div([
        html.Div([
            dcc.Graph(
                figure=px.scatter(df, x='Age', y='Total_Purchase', color='Churn',
                                  title="📈 Deep Dive: Customer Age vs. Total Purchase (Segmented by Churn)",
                                  template="plotly_dark",
                                  color_discrete_map={0: '#00ff88', '0': '#00ff88', 1: '#ff4b4b', '1': '#ff4b4b'}).update_layout(
                    plot_bgcolor='#111827', paper_bgcolor='#111827', font_color='#f3f4f6', transition_duration=500
                ) if not df.empty else {}
            )
        ], style={'backgroundColor': '#111827', 'padding': '20px', 'borderRadius': '12px', 'boxShadow': '0 10px 25px rgba(0,0,0,0.5)', 'border': '1px solid #1f2937'})
    ], style={'marginBottom': '40px'}),
    
    # Interactive Prediction Form Section
    html.Div([
        html.H3("🔮 Real-Time Customer Churn Prediction Engine", style={'color': '#00ff88', 'marginBottom': '15px', 'fontWeight': '600'}),
        html.P("Enter customer details below to instantly predict if they are at risk of churning.", style={'color': '#9ca3af', 'marginBottom': '25px'}),
        
        html.Div([
            html.Div([
                html.Label("Customer Age:", style={'color': '#f3f4f6', 'fontWeight': '500'}),
                dcc.Input(id='input-age', type='number', value=35, style={'width': '100%', 'padding': '10px', 'marginTop': '5px', 'backgroundColor': '#1f2937', 'color': '#fff', 'border': '1px solid #374151', 'borderRadius': '6px'})
            ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '3%'}),
            
            html.Div([
                html.Label("Total Purchase ($):", style={'color': '#f3f4f6', 'fontWeight': '500'}),
                dcc.Input(id='input-purchase', type='number', value=5000, style={'width': '100%', 'padding': '10px', 'marginTop': '5px', 'backgroundColor': '#1f2937', 'color': '#fff', 'border': '1px solid #374151', 'borderRadius': '6px'})
            ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '3%'}),
            
            html.Div([
                html.Label("Account Manager (0 or 1):", style={'color': '#f3f4f6', 'fontWeight': '500'}),
                dcc.Input(id='input-manager', type='number', value=1, style={'width': '100%', 'padding': '10px', 'marginTop': '5px', 'backgroundColor': '#1f2937', 'color': '#fff', 'border': '1px solid #374151', 'borderRadius': '6px'})
            ], style={'width': '30%', 'display': 'inline-block'})
        ], style={'marginBottom': '20px'}),
        
        html.Div([
            html.Div([
                html.Label("Years with Company:", style={'color': '#f3f4f6', 'fontWeight': '500'}),
                dcc.Input(id='input-years', type='number', value=2.5, style={'width': '100%', 'padding': '10px', 'marginTop': '5px', 'backgroundColor': '#1f2937', 'color': '#fff', 'border': '1px solid #374151', 'borderRadius': '6px'})
            ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '3%'}),
            
            html.Div([
                html.Label("Number of Sites:", style={'color': '#f3f4f6', 'fontWeight': '500'}),
                dcc.Input(id='input-sites', type='number', value=6, style={'width': '100%', 'padding': '10px', 'marginTop': '5px', 'backgroundColor': '#1f2937', 'color': '#fff', 'border': '1px solid #374151', 'borderRadius': '6px'})
            ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '3%'}),
            
            html.Div([
                html.Label("Location:", style={'color': '#f3f4f6', 'fontWeight': '500'}),
                dcc.Input(id='input-location', type='text', value='New York', style={'width': '100%', 'padding': '10px', 'marginTop': '5px', 'backgroundColor': '#1f2937', 'color': '#fff', 'border': '1px solid #374151', 'borderRadius': '6px'})
            ], style={'width': '30%', 'display': 'inline-block'})
        ], style={'marginBottom': '25px'}),
        
        html.Button("Run Prediction", id='predict-btn', n_clicks=0, style={
            'backgroundColor': '#00ff88', 'color': '#0b0f19', 'padding': '12px 25px', 'fontWeight': 'bold', 'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer', 'fontSize': '16px', 'boxShadow': '0 5px 15px rgba(0,255,136,0.3)'
        }),
        
        html.Div(id='prediction-output', style={'marginTop': '25px'})
        
    ], style={'backgroundColor': '#111827', 'padding': '30px', 'borderRadius': '12px', 'boxShadow': '0 10px 25px rgba(0,0,0,0.5)', 'border': '1px solid #1f2937', 'marginBottom': '40px'}),
    
    # Interactive Table Section
    html.Div([
        html.H3("📋 Executive Customer Records Explorer", style={'color': '#00ff88', 'marginBottom': '20px', 'fontWeight': '600'}),
        dash_table.DataTable(
            data=df.head(15).to_dict('records') if not df.empty else [],
            columns=[{"name": i, "id": i} for i in df.columns[:7]] if not df.empty else [],
            style_header={'backgroundColor': '#1f2937', 'color': '#00ff88', 'fontWeight': 'bold', 'border': 'none', 'padding': '12px'},
            style_cell={'backgroundColor': '#111827', 'color': '#f3f4f6', 'textAlign': 'left', 'border': '1px solid #1f2937', 'padding': '12px'},
            style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#1a2234'}],
            page_size=5
        )
    ], style={'backgroundColor': '#111827', 'padding': '25px', 'borderRadius': '12px', 'boxShadow': '0 10px 25px rgba(0,0,0,0.5)', 'border': '1px solid #1f2937'})
])

# Callback for Real-Time Prediction
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-btn', 'n_clicks')],
    [State('input-age', 'value'),
     State('input-purchase', 'value'),
     State('input-manager', 'value'),
     State('input-years', 'value'),
     State('input-sites', 'value'),
     State('input-location', 'value')]
)
def predict_customer_churn(n_clicks, age, purchase, manager, years, sites, location):
    if not n_clicks or n_clicks == 0:
        return html.Div("Click 'Run Prediction' above to see the real-time AI inference result.", style={'color': '#9ca3af', 'fontStyle': 'italic'})
    
    if model is None:
        return html.Div([
            html.H4("❌ Model Not Available", style={'color': '#ff4b4b', 'margin': '0 0 5px 0'}),
            html.P("Model could not be initialized.", style={'margin': '0', 'color': '#f3f4f6'})
        ], style={'backgroundColor': '#1f2937', 'borderLeft': '5px solid #ff4b4b', 'padding': '15px', 'borderRadius': '6px'})
    
    try:
        # Construct input DataFrame for prediction matching the trained features
        input_data = pd.DataFrame({
            'Age': [float(age) if age else 35.0],
            'Total_Purchase': [float(purchase) if purchase else 5000.0],
            'Account_Manager': [int(manager) if manager is not None else 1],
            'Years': [float(years) if years else 2.5],
            'Num_Sites': [float(sites) if sites else 6.0]
        })
        
        prediction = model.predict(input_data)[0]
        
        if prediction == 1:
            return html.Div([
                html.H4("⚠️ High Risk Warning (Will Churn)", style={'color': '#ff4b4b', 'margin': '0 0 5px 0'}),
                html.P("Prediction Result: This customer is **LIKELY TO CHURN** (Class 1). Immediate retention action required!", style={'margin': '0', 'color': '#f3f4f6'})
            ], style={'backgroundColor': '#1f2937', 'borderLeft': '5px solid #ff4b4b', 'padding': '15px', 'borderRadius': '6px'})
        else:
            return html.Div([
                html.H4("✅ Low Risk Status (Will Stay)", style={'color': '#00ff88', 'margin': '0 0 5px 0'}),
                html.P("Prediction Result: This customer is **LIKELY TO STAY** (Class 0). Customer is stable.", style={'margin': '0', 'color': '#f3f4f6'})
            ], style={'backgroundColor': '#1f2937', 'padding': '15px', 'borderRadius': '6px', 'borderLeft': '5px solid #00ff88'})
            
    except Exception as e:
        return html.Div(f"Error during prediction execution: {str(e)}", style={'color': '#ff4b4b', 'backgroundColor': '#1f2937', 'padding': '15px', 'borderRadius': '6px'})

if __name__ == '__main__':
    app.run(debug=True, port=8050)