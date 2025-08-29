# streamlit run app.py
import streamlit as st
import pandas as pd
import joblib
import os, json
import pydeck as pdk

# ------------------ CONFIG & THEME ------------------
st.set_page_config(page_title="Predicción Airbnb Madrid", page_icon="🏠", layout="wide")
st.markdown("""
<style>
.main .block-container {max-width: 1200px;}
h1 span.app-title {background: linear-gradient(90deg,#ff4b4b20,#1f77b420); padding:.15rem .6rem; border-radius:.5rem;}
.badge {display:inline-block; padding:.15rem .5rem; border-radius:.35rem; background:#eee; font-size:.85rem; margin-right:.35rem;}
.dataframe td, .dataframe th {padding:.25rem .5rem;}
</style>
""", unsafe_allow_html=True)

# ------------------ LOAD ARTIFACTS ------------------
@st.cache_resource
def load_artifacts():
    modelo = joblib.load("modelo_random_forest_airbnb.pkl")
    encoder = joblib.load("ordinal_encoder.pkl")
    return modelo, encoder

modelo, encoder = load_artifacts()
CAT_COLS = list(getattr(encoder, "feature_names_in_", []))

ROOM_TYPE_ES2EN = {
    "Apartamento entero": "Entire home/apt",
    "Habitación privada": "Private room",
    "Habitación compartida": "Shared room",
    "Habitación de hotel": "Hotel room",
}

# Buckets por rangos (cubre min=9, max=519 €)
PRICE_BUCKETS = [
    ("≤ 50 €",   0,   50),
    ("51-100 €", 51, 100),
    ("101-150 €",101, 150),
    ("151-200 €",151, 200),
    ("201-300 €",201, 300),
    ("301-400 €",301, 400),
    ("> 400 €",  401, None),
]

MONTH_ES = {1:"enero",2:"febrero",3:"marzo",4:"abril",5:"mayo",6:"junio",
            7:"julio",8:"agosto",9:"septiembre",10:"octubre",11:"noviembre",12:"diciembre"}

def _bucket_limits(bucket_name: str):
    for name, lo, hi in PRICE_BUCKETS:
        if name == bucket_name:
            return lo, hi
    return None, None

def fmt_eur_int(x) -> str:
    return f"{int(round(float(x)))} €"

def _default_review_block():
    return dict(
        review_scores_rating=95, review_scores_accuracy=9, review_scores_cleanliness=9,
        review_scores_checkin=9, review_scores_communication=9, review_scores_location=9,
        review_scores_value=9, number_of_reviews=30, host_is_superhost=0,
        host_response_time=1, host_response_rate=100.0, host_acceptance_rate=100.0,
        host_identity_verified=0, availability_365=180, minimum_nights=2,
        instant_bookable=0, precio_promedio_mensual=0
    )

def barrios_modelo():
    enc_feats = list(getattr(encoder, "feature_names_in_", []))
    if "neighbourhood_cleansed" in enc_feats:
        i = enc_feats.index("neighbourhood_cleansed")
        return list(encoder.categories_[i])
    return []

# --------- Reglas de plausibilidad ---------
def reglas_por_tipo(tipo):
    if tipo == "Apartamento entero":
        return dict(per=(1,16), hab=(0,10), ban=(1,10))
    if tipo == "Habitación privada":
        return dict(per=(1,4), hab=(1,1), ban=(0,1))
    if tipo == "Habitación compartida":
        return dict(per=(1,6), hab=(1,1), ban=(0,2))
    # Habitación de hotel
    return dict(per=(1,4), hab=(1,1), ban=(1,2))

def comprobar_plausibilidad(tipo, personas, habitaciones, banos):
    msgs = []
    limites = reglas_por_tipo(tipo)
    per_min, per_max = limites["per"]
    hab_min, hab_max = limites["hab"]
    ban_min, ban_max = limites["ban"]

    per_ok = min(max(personas, per_min), per_max)
    hab_ok = min(max(habitaciones, hab_min), hab_max)
    ban_ok = min(max(banos, ban_min), ban_max)

    if tipo == "Apartamento entero":
        ban_top = max(1, min(5, hab_ok + 1))
        if ban_ok > ban_top:
            msgs.append(f"Para {hab_ok} habitación(es), sugerimos como máximo **{ban_top} baño(s)**.")
            ban_ok = ban_top
        per_top = max(1, 2*hab_ok + 2)
        if per_ok > per_top:
            msgs.append(f"Con {hab_ok} habitación(es), el máximo razonable es **{per_top}** personas.")
            per_ok = per_top

    changed = (per_ok != personas) or (hab_ok != habitaciones) or (ban_ok != banos)
    return dict(valid=not changed, personas=per_ok, habitaciones=hab_ok, banos=ban_ok, mensajes=msgs)

# --------- Core de predicción ---------
def _build_base_df(barrio, room_type_es, accommodates, bedrooms, bathrooms, mes):
    room_type = ROOM_TYPE_ES2EN[room_type_es]
    if room_type_es in {"Habitación privada", "Habitación compartida", "Habitación de hotel"}:
        bedrooms = 1
    row = dict(
        neighbourhood_cleansed=barrio, room_type=room_type, accommodates=accommodates,
        bedrooms=bedrooms, beds=bedrooms if bedrooms is not None else 1,
        bathrooms=bathrooms, mes=mes,
    )
    row.update(_default_review_block())
    df = pd.DataFrame([row])
    cols_to_encode = [c for c in df.columns if c in CAT_COLS]
    if cols_to_encode:
        df[cols_to_encode] = encoder.transform(df[cols_to_encode])
    model_cols = list(getattr(modelo, "feature_names_in_", []))
    if model_cols:
        for missing in set(model_cols) - set(df.columns):
            df[missing] = 0
        df = df.reindex(columns=model_cols, fill_value=0)
    return df

def predecir_precio(barrio, room_type_es, accommodates, bedrooms, bathrooms, mes):
    df = _build_base_df(barrio, room_type_es, accommodates, bedrooms, bathrooms, mes)
    return float(modelo.predict(df)[0])

def comparativa_meses(barrio, room_type_es, accommodates, bedrooms, bathrooms, mes, meses_extra):
    registros = [{"mes": mes, "precio": predecir_precio(barrio, room_type_es, accommodates, bedrooms, bathrooms, mes), "seleccionado": True}]
    for m in meses_extra:
        registros.append({"mes": m, "precio": predecir_precio(barrio, room_type_es, accommodates, bedrooms, bathrooms, m), "seleccionado": False})
    return pd.DataFrame(registros).sort_values(["seleccionado","mes"], ascending=[False,True]).reset_index(drop=True)

def alternativas_barrios_raw(barrio_actual, room_type_es, accommodates, bedrooms, bathrooms, mes):
    lista = [b for b in barrios_modelo() if b != barrio_actual]
    if not lista:
        return pd.DataFrame(columns=["barrio","precio","delta","diff_abs"]), None
    precio_ref = predecir_precio(barrio_actual, room_type_es, accommodates, bedrooms, bathrooms, mes)
    filas = []
    for b in lista:
        p = predecir_precio(b, room_type_es, accommodates, bedrooms, bathrooms, mes)
        filas.append({"barrio": b, "precio": p, "delta": p - precio_ref, "diff_abs": abs(p - precio_ref)})
    return pd.DataFrame(filas), precio_ref

# ===================== MAPA DE BARRIOS =====================
def mostrar_mapa_barrios(barrio_elegido, cercanos_df=None, baratos_df=None,
                         geojson_path="../datos_brutos/neighbourhoods.geojson"):
    """
    Pinta un mapa de Madrid con:
      - barrio_elegido en rojo
      - alternativos (cercanos/baratos) en verde (máx 2 y 2)
      - resto en gris
    """
    def _norm(x): return str(x).strip().casefold() if x is not None else ""

    # 1) Cargar GeoJSON
    if not os.path.exists(geojson_path):
        st.error(f"🗺️ No encuentro el GeoJSON en: {os.path.abspath(geojson_path)}")
        return

    try:
        with open(geojson_path, "r", encoding="utf-8") as f:
            gj = json.load(f)
        features = gj.get("features", [])
        if not features:
            st.error("El GeoJSON no trae 'features'.")
            return
    except Exception as e:
        st.error("Error leyendo el GeoJSON.")
        st.exception(e)
        return

    # 2) Detectar la clave del nombre en properties
    name_key = None
    for k in ("neighbourhood", "neighborhood", "name", "NAME", "NOMBRE"):
        if k in features[0].get("properties", {}):
            name_key = k
            break
    if not name_key:
        st.error("No pude detectar la clave del nombre del barrio en el GeoJSON.")
        return

    # 3) Preparar lista de barrios destacados
    destacados = {}
    if barrio_elegido is not None:
        destacados[_norm(barrio_elegido)] = "elegido"

    alt_cerc = list(cercanos_df["Barrio"]) if (cercanos_df is not None and len(cercanos_df)) else []
    alt_bar  = list(baratos_df["Barrio"])  if (baratos_df  is not None and len(baratos_df))  else []
    for b in (alt_cerc[:2] + alt_bar[:2]):
        destacados[_norm(b)] = "alternativo"

    # 4) Colores
    COL_ELEGIDO     = [220, 60, 60, 160]   # rojo
    COL_ALTERNATIVO = [ 60,160, 90, 160]   # verde
    COL_RESTO       = [170,170,170, 40]    # gris

    # 5) Enriquecer features con color/etiqueta
    feats_col = []
    for ft in features:
        props  = dict(ft.get("properties", {}))
        nombre = props.get(name_key)
        clave  = _norm(nombre)

        if clave in destacados:
            fill   = COL_ELEGIDO if destacados[clave] == "elegido" else COL_ALTERNATIVO
            stroke = [60, 60, 60]; width = 2
        else:
            fill   = COL_RESTO
            stroke = [120,120,120]; width = 1

        props.update({
            "fill_r": int(fill[0]), "fill_g": int(fill[1]), "fill_b": int(fill[2]), "fill_a": int(fill[3]),
            "stroke_r": int(stroke[0]), "stroke_g": int(stroke[1]), "stroke_b": int(stroke[2]),
            "line_w": int(width),
            "label": nombre if nombre is not None else "(sin nombre)"
        })
        ft2 = dict(ft)
        ft2["properties"] = props
        feats_col.append(ft2)

    # 6) Capa y render
    layer = pdk.Layer(
        "GeoJsonLayer",
        data={"type": "FeatureCollection", "features": feats_col},
        pickable=True,
        stroked=True,
        filled=True,
        get_fill_color="[properties.fill_r, properties.fill_g, properties.fill_b, properties.fill_a]",
        get_line_color="[properties.stroke_r, properties.stroke_g, properties.stroke_b]",
        get_line_width="properties.line_w",
        lineWidthMinPixels=1,
        auto_highlight=True,
    )
    view_state = pdk.ViewState(latitude=40.4168, longitude=-3.7038, zoom=10.5)

    st.subheader("Mapa de barrios")
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{label}"}))
# ===================== FIN MAPA =====================

# ------------------ HEADER ------------------
colh1, colh2 = st.columns([1,4])
with colh1:
    try: st.image("logo.png", width=80)
    except: st.write("")
with colh2:
    st.markdown('<h1><span class="app-title">Predicción de precio • Airbnb Madrid</span></h1>', unsafe_allow_html=True)
    st.markdown(
    """
    <p style='text-align: center; font-size:1rem; color: #555; margin-top:-10px;'>
        Predice, compara y encuentra el precio justo en <b>Airbnb Madrid</b>
    </p>
    """,
    unsafe_allow_html=True
)

# ------------------ SIDEBAR (inputs) ------------------
with st.sidebar:
    st.markdown("**Ajustes**")
    barrios = barrios_modelo()
    barrio_elegido = st.selectbox("Barrio", barrios if barrios else ["Centro"])
    tipo_aloj_es = st.selectbox("Tipo de alojamiento", list(ROOM_TYPE_ES2EN.keys()))

    lim = reglas_por_tipo(tipo_aloj_es)
    per_min, per_max = lim["per"]
    hab_min, hab_max = lim["hab"]
    ban_min, ban_max = lim["ban"]

    personas = st.number_input("Personas", min_value=per_min, max_value=per_max, value=min(2, per_max), step=1)
    if hab_min == hab_max:
        st.write(f"🔒 Habitaciones: **{hab_min}** (fijo para {tipo_aloj_es.lower()}).")
        habitaciones = hab_min
    else:
        habitaciones = st.number_input("Habitaciones", min_value=hab_min, max_value=hab_max, value=1, step=1)
    banos = st.number_input("Baños", min_value=ban_min, max_value=ban_max, value=max(1, ban_min), step=1)

    mes_elegido = st.selectbox("Mes", list(range(1,13)), index=6, format_func=lambda m: MONTH_ES[m])
    comparar = st.checkbox("Comparar con otros meses", value=True)
    meses_opciones = [m for m in range(1,13) if m != mes_elegido]
    meses_default  = [m for m in [1,3,6,9,12] if m != mes_elegido]
    meses_comparar = st.multiselect("Meses a comparar", meses_opciones, default=meses_default,
                                    format_func=lambda m: MONTH_ES[m]) if comparar else []
    comparar_todos = st.checkbox("Comparar con TODOS los meses", value=False) if comparar else False
    if comparar and comparar_todos:
        meses_comparar = meses_opciones

    rango_precio = st.selectbox("Rango de precio deseado", [n for n,_,_ in PRICE_BUCKETS])
    filtrar_alt = st.checkbox("Filtrar alternativas por mi rango de precio", value=False)

# ------------------ VALIDACIÓN UX (antes de predecir) ------------------
chk = comprobar_plausibilidad(tipo_aloj_es, personas, habitaciones, banos)
if not chk["valid"]:
    st.error("No disponemos de alojamientos con esa combinación exacta.")
    for m in chk["mensajes"]:
        st.write("• " + m)
    st.info(f"**Sugerencia** → Personas: {chk['personas']} | Habitaciones: {chk['habitaciones']} | Baños: {chk['banos']}")
else:
    # ------------------ UNA SOLA PANTALLA: dos columnas ------------------
    col1, col2 = st.columns([1,1])

    with col1:
        precio_mes = predecir_precio(barrio_elegido, tipo_aloj_es, personas, habitaciones, banos, mes_elegido)
        st.metric(f"**Precio estimado en {MONTH_ES[mes_elegido]}**", fmt_eur_int(precio_mes))

        df_comp = comparativa_meses(barrio_elegido, tipo_aloj_es, personas, habitaciones, banos, mes_elegido, meses_comparar)
        df_comp['Mes'] = df_comp['mes'].map(MONTH_ES)
        df_comp['Precio (estimado)'] = df_comp['precio'].map(fmt_eur_int)
        st.subheader("Comparativa por meses")
        st.dataframe(df_comp[['Mes','Precio (estimado)','seleccionado']], use_container_width=True, hide_index=True)

        lo, hi = _bucket_limits(rango_precio)
        if lo is not None:
            df_tmp = df_comp.copy()
            df_tmp['precio_int'] = df_tmp['precio'].round(0).astype(int)
            en_rango = df_tmp[(df_tmp['precio_int'] >= lo) & ((hi is None) | (df_tmp['precio_int'] <= hi))]
            if len(en_rango) == 0:
                target = (lo + (hi if hi is not None else lo)) / 2
                idx = (df_tmp['precio_int'] - target).abs().idxmin()
                st.info(f"Ningún mes cae en {rango_precio}. Sugerencia: **{df_tmp.loc[idx,'Mes']}** (~{fmt_eur_int(df_tmp.loc[idx,'precio'])}).")
            else:
                meses_ok = ", ".join(en_rango['Mes'].tolist())
                st.success(f"Meses dentro del rango {rango_precio}: {meses_ok}")

        with st.expander("Mín/Máx estimado (estos meses)"):
            if len(df_comp):
                pmin = int(df_comp['precio'].min().round())
                pmax = int(df_comp['precio'].max().round())
                st.markdown(f"<span class='badge'>Mín: {pmin} €</span> <span class='badge'>Máx: {pmax} €</span>", unsafe_allow_html=True)

    with col2:
        st.subheader("Alternativas por barrio")
        df_alt_raw, precio_ref = alternativas_barrios_raw(barrio_elegido, tipo_aloj_es, personas, habitaciones, banos, mes_elegido)

        # Filtrado por rango (si se pide)
        if filtrar_alt and len(df_alt_raw):
            lo, hi = _bucket_limits(rango_precio)
            if lo is not None:
                df_alt_raw['precio_int'] = df_alt_raw['precio'].round(0).astype(int)
                df_alt_raw = df_alt_raw[(df_alt_raw['precio_int'] >= lo) & ((hi is None) | (df_alt_raw['precio_int'] <= hi))]

        # Inicializamos por si no hay alternativas (así el mapa no falla)
        cercanos = None
        baratos  = None

        if len(df_alt_raw):
            cercanos = df_alt_raw.sort_values('diff_abs').head(2).copy()
            baratos  = df_alt_raw.sort_values(['precio','diff_abs']).head(2).copy()

            for df_ in (cercanos, baratos):
                df_['Precio (estimado)'] = df_['precio'].map(fmt_eur_int)
                df_['Δ vs elegido'] = df_['delta'].map(fmt_eur_int).map(lambda s: "+"+s if not s.startswith("-") else s)
                df_.rename(columns={'barrio':'Barrio'}, inplace=True)

            st.markdown("**Precios más cercanos al barrio elegido**")
            st.dataframe(cercanos[['Barrio','Precio (estimado)','Δ vs elegido']], use_container_width=True, hide_index=True)

            st.markdown("**Alternativas más baratas**")
            st.dataframe(baratos[['Barrio','Precio (estimado)','Δ vs elegido']], use_container_width=True, hide_index=True)
        else:
            st.info("No hay alternativas tras aplicar el filtro de rango.")

        resumen = (
            f"Barrio: {barrio_elegido} | Tipo: {tipo_aloj_es} | Personas: {personas} | "
            f"Habitaciones: {habitaciones} | Baños: {banos} | Mes: {MONTH_ES[mes_elegido]} | "
            f"Precio estimado: {fmt_eur_int(precio_mes)}"
        )
        st.markdown("**Resumen de busqueda**")
        st.text_area("Tus preferencias:", value=resumen, height=80)

    # ---------- MAPA (abajo, a todo el ancho) ----------
    mostrar_mapa_barrios(
        barrio_elegido=barrio_elegido,
        cercanos_df=cercanos,
        baratos_df=baratos,
        geojson_path="../datos_brutos/neighbourhoods.geojson"
    )