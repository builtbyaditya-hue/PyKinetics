import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.integrate import solve_ivp

# Import our physics rules
from constants import (LAMBDA_PROMPT, BETA_TOTAL, BETA_I, LAMBDA_I, 
                       ALPHA_T, HEATING_RATE, COOLING_RATE, T_INITIAL)

# --- 1. Global Simulation Memory ---
MAX_POINTS = 60 # Keeps the graph light and fast
t_history = [0.0]
p_history = [1.0]
temp_history = [T_INITIAL]

C0 = (BETA_I / (LAMBDA_PROMPT * LAMBDA_I)) * 1.0
current_state = [1.0] + C0.tolist() + [T_INITIAL]
current_time = 0.0

# --- 2. The Physics Step Function ---
def point_kinetics_equations(t, y, reactivity_dollars, coolant_factor):
    n, C, T = y[0], y[1:7], y[7]
    rho_abs = (reactivity_dollars + ALPHA_T * (T - T_INITIAL)) * BETA_TOTAL
    dn_dt = ((rho_abs - BETA_TOTAL) / LAMBDA_PROMPT) * n + np.sum(LAMBDA_I * C)
    dC_dt = (BETA_I / LAMBDA_PROMPT) * n - LAMBDA_I * C
    # Apply the coolant slider to the cooling rate (63% is our baseline 1.0x rate)
    active_cooling = COOLING_RATE * (coolant_factor / 63.0)
    dT_dt = HEATING_RATE * (n - 1.0) - active_cooling * (T - T_INITIAL)
    return [dn_dt] + dC_dt.tolist() + [dT_dt]

# --- 3. Build the Dash App UI ---
app = dash.Dash(__name__)

# Color Palette matching your target image
BG_COLOR = "#0b0f19"
CARD_BG = "#151b28"
TEXT_COLOR = "#e2e8f0"

app.layout = html.Div(style={'backgroundColor': BG_COLOR, 'color': TEXT_COLOR, 'fontFamily': 'Segoe UI, sans-serif', 'padding': '2rem', 'minHeight': '100vh'}, children=[
    
    # Header
    html.Div([
        html.H2("Nuclear Core Simulation", style={'margin': '0', 'fontWeight': '400'}),
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '20px'}),

    # The Visual Core Graph
    html.Div([
        dcc.Graph(id='core-visualizer', config={'displayModeBar': False})
    ], style={'backgroundColor': CARD_BG, 'padding': '10px', 'borderRadius': '8px', 'marginBottom': '20px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.3)'}),
    
    # The Live Ticker Graph
    html.Div([
        html.Div("Metrics ↑", style={'fontSize': '12px', 'color': '#94a3b8', 'marginBottom': '5px'}),
        dcc.Graph(id='live-update-graph', config={'displayModeBar': False})
    ], style={'marginBottom': '30px'}),
    
    # Metrics Panel
    html.Div([
        html.Div([
            html.Div("NEUTRON FLUX", style={'fontSize': '12px', 'color': '#94a3b8', 'fontWeight': 'bold'}),
            html.Div(id='metric-flux', style={'fontSize': '24px', 'fontWeight': 'bold'})
        ], style={'textAlign': 'center', 'flex': '1', 'borderRight': '1px solid #334155'}),
        
        html.Div([
            html.Div("CORE TEMP", style={'fontSize': '12px', 'color': '#94a3b8', 'fontWeight': 'bold'}),
            html.Div(id='metric-temp', style={'fontSize': '24px', 'fontWeight': 'bold'})
        ], style={'textAlign': 'center', 'flex': '1', 'borderRight': '1px solid #334155'}),
        
        html.Div([
            html.Div("STATUS", style={'fontSize': '12px', 'color': '#94a3b8', 'fontWeight': 'bold'}),
            html.Div(id='metric-status', style={'fontSize': '20px', 'fontWeight': 'bold'})
        ], style={'textAlign': 'center', 'flex': '1'})
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'padding': '20px', 'backgroundColor': CARD_BG, 'borderRadius': '8px', 'marginBottom': '30px'}),

    # Sliders Panel
    html.Div([
        html.Div([
            html.Label("Control Rods (%)", style={'marginRight': '15px', 'fontWeight': 'bold'}),
            html.Div(dcc.Slider(id='rod-slider', min=0, max=100, step=1, value=88, tooltip={"placement": "top"}), style={'flex': '1'}),
        ], style={'display': 'flex', 'alignItems': 'center', 'flex': '1', 'marginRight': '20px'}),
        
        html.Div([
            html.Label("Coolant Flow (%)", style={'marginRight': '15px', 'fontWeight': 'bold'}),
            html.Div(dcc.Slider(id='coolant-slider', min=0, max=100, step=1, value=63, tooltip={"placement": "top"}), style={'flex': '1'}),
        ], style={'display': 'flex', 'alignItems': 'center', 'flex': '1'})
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'padding': '0 20px'}),

    # The High-Speed Timer (150ms for smooth animation)
    dcc.Interval(id='interval-component', interval=150, n_intervals=0)
])


# --- 4. The Master Callback ---
@app.callback(
    [Output('core-visualizer', 'figure'),
     Output('live-update-graph', 'figure'),
     Output('metric-flux', 'children'),
     Output('metric-temp', 'children'),
     Output('metric-status', 'children'),
     Output('metric-status', 'style')],
    [Input('interval-component', 'n_intervals')],
    [State('rod-slider', 'value'), State('coolant-slider', 'value')]
)
def update_dashboard(n, rod_val, coolant_val):
    global current_time, current_state, t_history, p_history, temp_history
    
    # 1. Math Step
    dollars = (rod_val / 100.0) * 0.90 
    t_span = (current_time, current_time + 0.15)
    
    # ADD rtol and atol HERE to stop the numerical zig-zagging
    sol = solve_ivp(
        point_kinetics_equations, t_span, current_state, 
        args=(dollars, coolant_val), method='BDF',
        rtol=1e-6, atol=1e-9
    )
    
    current_time = t_span[1]
    current_state = sol.y[:, -1]
    
    t_history.append(current_time)
    p_history.append(current_state[0])
    temp_history.append(current_state[7])
    
    if len(t_history) > MAX_POINTS:
        t_history.pop(0)
        p_history.pop(0)
        temp_history.pop(0)

    latest_p = p_history[-1]
    latest_t = temp_history[-1]

    # 2. Draw Core Visualizer
    core_fig = go.Figure()
    core_fig.add_trace(go.Bar(
        x=[1, 2, 3, 4, 5], y=[rod_val]*5, 
        width=0.2, marker_color="#cbd5e1", hoverinfo='none'
    ))
    glow = 0.1 + (rod_val / 200.0)
    core_fig.update_layout(
        plot_bgcolor=f"rgba(220, 38, 38, {glow})", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, range=[0, 6]), yaxis=dict(visible=False, range=[0, 100]),
        height=200, margin=dict(l=0, r=0, t=0, b=0), uirevision=True # <--- UI REVISION PREVENTS LAG
    )

   # 3. Draw Live Line Graph
    line_fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # ADD shape='spline' to both lines here!
    line_fig.add_trace(go.Scatter(
        x=t_history, y=temp_history, name="Core Temp", 
        line=dict(color='#3b82f6', width=3, shape='spline')
    ), secondary_y=True)
    
    line_fig.add_trace(go.Scatter(
        x=t_history, y=p_history, name="Neutron Flux", 
        line=dict(color='#22c55e', width=3, shape='spline')
    ), secondary_y=False)
    line_fig.add_hline(y=1.8, line_dash="dash", line_color="#ef4444", annotation_text="Critical Hazard Limit", secondary_y=False)
    
    line_fig.update_layout(
        template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=20, b=20), height=300, showlegend=False,
        xaxis=dict(range=[min(t_history), max(t_history)], showgrid=False),
        yaxis=dict(range=[0, max(2.0, max(p_history) * 1.1)], showgrid=True, gridcolor='#334155'),
        yaxis2=dict(range=[300, max(360, max(temp_history) * 1.05)], showgrid=False),
        uirevision=True # <--- UI REVISION PREVENTS LAG
    )

    # 4. Metrics Logic
    flux_text = f"{latest_p:.0f}" if latest_p > 10 else f"{latest_p:.2f}"
    temp_text = f"{latest_t:.0f}°C"
    
    if latest_t > 340:
        status_text = "High Temp Alert"
        status_style = {'fontSize': '20px', 'fontWeight': 'bold', 'color': '#ef4444'}
    else:
        status_text = "Nominal"
        status_style = {'fontSize': '20px', 'fontWeight': 'bold', 'color': '#22c55e'}

    return core_fig, line_fig, flux_text, temp_text, status_text, status_style

if __name__ == '__main__':
    app.run(debug=True)