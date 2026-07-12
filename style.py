"""
style.py
Injects DocIQ's visual theme into Streamlit.

Design language v2 -- "Studio": a clean, monochrome-blue SaaS look inspired
by modern chat/RAG products. Fixed dark-navy sidebar (brand rail) paired
with a main content area that switches between light and dark. The upload
flow is rebuilt around a prominent, native drag-and-drop dropzone (real
browser drag & drop -- Streamlit's file uploader already supports it, this
just makes it look and feel like the centerpiece of the app instead of a
buried form control).

Every widget Streamlit renders -- including the ones that live outside the
normal DOM tree, like selectbox dropdowns and the file-uploader dropzone --
is explicitly themed so dark mode never produces invisible or low-contrast
text.
"""


# The sidebar is a fixed dark "brand rail" in BOTH themes (like Notion/
# Linear/most modern chat apps) -- only the main content area switches
# between light and dark. This mirrors the reference design.
SIDEBAR = {
    "bg": "#11121a",
    "bg_soft": "#171826",
    "border": "#242637",
    "text": "#95a1e0",
    "muted": "#9497ab",
    "faint": "#8b90ab",
    "active": "#1c1e2e",
    "active_border": "#3b6ef6",
}

THEMES = {
    "dark": {
        "bg": "#0c0d13",
        "bg_secondary": "#12131c",
        "card_bg": "#171822",
        "card_hover": "#1e202c",
        "text": "#95a1e0",
        "text_muted": "#a6a9bd",
        "text_faint": "#71748a",
        "border": "#262838",
        "border_soft": "#1d1e2b",
        "input_bg": "#14151f",
        "shadow": "0 12px 32px rgba(0, 0, 0, 0.45)",
        "dropzone_bg": "#101827",
        "dropzone_bg_hover": "#132038",
    },
    "light": {
        "bg": "#f5f6fa",
        "bg_secondary": "#ffffff",
        "card_bg": "#ffffff",
        "card_hover": "#f0f2fa",
        "text": "#14151f",
        "text_muted": "#5c5f74",
        "text_faint": "#8b8ea3",
        "border": "#e6e8f2",
        "border_soft": "#eef0f8",
        "input_bg": "#ffffff",
        "shadow": "0 10px 28px rgba(20, 30, 70, 0.08)",
        "dropzone_bg": "#eef4ff",
        "dropzone_bg_hover": "#e2ecff",
    },
}

# Monochrome-blue accent system -- shared across both themes.
ACCENT = "#3b6ef6"
ACCENT_LIGHT = "#7fa6ff"
ACCENT_DARK = "#2952c8"
BEACON = "#3b6ef6"          # kept as a distinct name for the "send" action;
BEACON_LIGHT = "#7fa6ff"    # same blue family so the palette stays monochrome
DANGER = {"dark": "#f87171", "light": "#dc2626"}
SUCCESS = {"dark": "#4ade80", "light": "#16a34a"}


def get_css(dark: bool) -> str:
    key = "dark" if dark else "light"
    theme = THEMES[key]

    bg = theme["bg"]
    bg_secondary = theme["bg_secondary"]
    card_bg = theme["card_bg"]
    card_hover = theme["card_hover"]
    text = theme["text"]
    text_muted = theme["text_muted"]
    text_faint = theme["text_faint"]
    border = theme["border"]
    border_soft = theme["border_soft"]
    input_bg = theme["input_bg"]
    shadow = theme["shadow"]
    dropzone_bg = theme["dropzone_bg"]
    dropzone_bg_hover = theme["dropzone_bg_hover"]
    danger = DANGER[key]
    success = SUCCESS[key]

    sb_bg = SIDEBAR["bg"]
    sb_bg_soft = SIDEBAR["bg_soft"]
    sb_border = SIDEBAR["border"]
    sb_text = SIDEBAR["text"]
    sb_muted = SIDEBAR["muted"]
    sb_faint = SIDEBAR["faint"]
    sb_active = SIDEBAR["active"]
    sb_active_border = SIDEBAR["active_border"]


    return f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {{
        --di-bg: {bg};
        --di-bg-secondary: {bg_secondary};
        --di-card: {card_bg};
        --di-card-hover: {card_hover};
        --di-text: {text};
        --di-muted: {text_muted};
        --di-faint: {text_faint};
        --di-border: {border};
        --di-border-soft: {border_soft};
        --di-input: {input_bg};
        --di-accent: {ACCENT};
        --di-accent-light: {ACCENT_LIGHT};
        --di-accent-dark: {ACCENT_DARK};
        --di-beacon: {BEACON};
        --di-beacon-light: {BEACON_LIGHT};
        --di-danger: {danger};
        --di-success: {success};
        --di-shadow: {shadow};
        --di-dropzone-bg: {dropzone_bg};
        --di-dropzone-bg-hover: {dropzone_bg_hover};

        --di-sidebar: {sb_bg};
        --di-sidebar-soft: {sb_bg_soft};
        --di-sidebar-border: {sb_border};
        --di-sidebar-text: {sb_text};
        --di-sidebar-muted: {sb_muted};
        --di-sidebar-faint: {sb_faint};
        --di-sidebar-active: {sb_active};
        --di-sidebar-active-border: {sb_active_border};

        --di-font-display: 'Space Grotesk', 'Inter', sans-serif;
        --di-font-body: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        --di-font-mono: 'JetBrains Mono', 'SFMono-Regular', Consolas, monospace;
    }}

    /* ------------------------------------------------------------------ */
    /* Base                                                                */
    /* ------------------------------------------------------------------ */

    .stApp {{
        background: var(--di-bg);
        color: var(--di-text);
        font-family: var(--di-font-body);
    }}

    body, p, span, div, li, label {{
        color: var(--di-text);
    }}

    h1, h2, h3, h4, h5, h6 {{
        font-family: var(--di-font-display) !important;
        color: var(--di-text) !important;
        letter-spacing: -0.01em;
    }}

    a {{ color: var(--di-accent-light); }}
    a:hover {{ color: var(--di-accent); }}

    ::selection {{ background: var(--di-accent); color: #ffffff; }}

    ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{
        background: var(--di-border);
        border-radius: 10px;
        border: 2px solid var(--di-bg);
    }}
    ::-webkit-scrollbar-thumb:hover {{ background: var(--di-accent-dark); }}

    button:focus-visible, input:focus-visible, textarea:focus-visible,
    [data-baseweb="select"]:focus-within, a:focus-visible {{
        outline: 2px solid var(--di-accent-light) !important;
        outline-offset: 2px;
    }}

    #MainMenu, footer, header[data-testid="stHeader"] {{ background: transparent; }}

    .block-container {{ padding-top: 1.6rem; max-width: 880px; }}

    /* ------------------------------------------------------------------ */
    /* Sidebar shell -- fixed dark "brand rail" regardless of theme        */
    /* ------------------------------------------------------------------ */

    [data-testid="stSidebar"] {{
        background: var(--di-sidebar);
        border-right: 1px solid var(--di-sidebar-border);
    }}

    [data-testid="stSidebar"] {{
        color: var(--di-sidebar-text) !important;
    }}

    [data-testid="stSidebar"] ::-webkit-scrollbar-thumb {{
        background: var(--di-sidebar-border);
        border: 2px solid var(--di-sidebar);
    }}
    [data-testid="stSidebar"] ::-webkit-scrollbar-thumb:hover {{
        background: var(--di-accent);
    }}

    [data-testid="stSidebar"] * {{
        color: var(--di-sidebar-text) !important;
    }}

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        color: var(--di-sidebar-text) !important;
        line-height: 1.5;
    }}

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
        color: var(--di-sidebar-faint) !important;
    }}

    .di-brand {{
        display: flex;
        align-items: center;
        gap: 10px;
        font-family: var(--di-font-display);
        font-size: 1.4rem;
        font-weight: 700;
        padding: 2px 0 4px 0;
        color: var(--di-sidebar-text);
    }}

    .di-brand-mark {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 30px;
        height: 30px;
        border-radius: 9px;
        background: linear-gradient(135deg, var(--di-accent-light), var(--di-accent-dark));
        font-size: 0.95rem;
    }}

    .di-signed-in {{
        color: var(--di-sidebar-faint) !important;
        font-size: 0.78rem;
        margin: -6px 0 6px 0;
    }}

    /* ------------------------------------------------------------------ */
    /* Sidebar section labels + cards                                      */
    /* ------------------------------------------------------------------ */

    .di-date-label {{
        font-family: var(--di-font-mono);
        font-size: 0.68rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--di-sidebar-faint) !important;
        margin: 18px 0 8px 4px;
    }}

    .di-doc-card {{
        background: var(--di-sidebar-soft);
        border: 1px solid var(--di-sidebar-border);
        border-radius: 12px;
        padding: 10px 12px;
        margin-bottom: 8px;
        transition: border-color 0.15s ease, background 0.15s ease, transform 0.15s ease, box-shadow 0.15s ease;
    }}

    .di-doc-card:hover {{
        border-color: var(--di-accent);
        background: var(--di-sidebar-active);
        transform: translateX(2px);
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3);
    }}

    .di-doc-card--active {{
        border-color: var(--di-sidebar-active-border);
        background: var(--di-sidebar-active);
        box-shadow: inset 2px 0 0 var(--di-sidebar-active-border);
    }}

    [data-testid="stSidebar"] .di-meta {{
        color: var(--di-sidebar-faint) !important;
    }}

    [data-testid="stSidebar"] .stButton > button {{
        text-align: left;
        justify-content: flex-start;
        border-left: 2px solid transparent;
        font-weight: 400;
        background: var(--di-sidebar-soft);
        border: 1px solid var(--di-sidebar-border);
        padding: 8px 12px;
        cursor: pointer;
        transition: border-color 0.15s ease, background 0.15s ease, color 0.15s ease, transform 0.12s ease;
    }}

    [data-testid="stSidebar"] .stButton > button:hover {{
        border-color: var(--di-accent);
        background: var(--di-sidebar-active);
        color: var(--di-accent-light) !important;
        transform: translateX(2px);
    }}

    [data-testid="stSidebar"] .stButton > button:active {{
        transform: translateX(2px) scale(0.98);
        background: var(--di-sidebar-active);
    }}

    [data-testid="stSidebar"] .stButton > button p {{ color: var(--di-sidebar-text) !important; }}
    [data-testid="stSidebar"] .stButton > button:hover p {{ color: var(--di-accent-light) !important; }}

    /* Document library column alignment */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {{
        align-items: center;
        display: flex;
        gap: 8px;
    }}

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
        display: flex;
        align-items: center;
        min-height: 44px;
    }}

    /* ------------------------------------------------------------------ */
    /* Hero (empty state, before any documents exist)                      */
    /* ------------------------------------------------------------------ */

    .di-hero {{
        text-align: center;
        padding: 28px 20px 8px 20px;
    }}

    .di-hero-mark {{
        width: 60px;
        height: 60px;
        margin: 0 auto 18px auto;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.7rem;
        background: linear-gradient(135deg, var(--di-accent-light), var(--di-accent-dark));
        box-shadow: 0 10px 30px {ACCENT}33;
    }}

    .di-hero-mark-sm {{
        width: 40px;
        height: 40px;
        margin: 0 auto 10px auto;
        border-radius: 12px;
        font-size: 1.15rem;
        box-shadow: 0 6px 18px {ACCENT}33;
    }}

    .di-hero-title {{
        font-family: var(--di-font-display);
        font-size: 2rem;
        font-weight: 700;
        color: var(--di-text);
        line-height: 1.25;
    }}

    .di-hero-sub {{
        font-size: 1.05rem;
        font-weight: 450;
        color: var(--di-muted);
        margin-top: 6px;
    }}

    /* ------------------------------------------------------------------ */
    /* Drag & drop dropzone -- Streamlit's native file uploader, restyled  */
    /* into the centerpiece upload card. Real browser drag-and-drop.       */
    /* ------------------------------------------------------------------ */

    [data-testid="stFileUploader"] {{
        width: 100%;
    }}

    [data-testid="stFileUploaderDropzone"] {{
        background: var(--di-dropzone-bg) !important;
        border: 2px dashed var(--di-accent-light) !important;
        border-radius: 20px !important;
        padding: 40px 24px !important;
        transition: border-color 0.15s ease, background 0.15s ease, transform 0.15s ease;
    }}

    [data-testid="stFileUploaderDropzone"]:hover {{
        border-color: var(--di-accent) !important;
        background: var(--di-dropzone-bg-hover) !important;
    }}

    [data-testid="stFileUploaderDropzoneInstructions"] {{
        color: var(--di-muted) !important;
        font-family: var(--di-font-body);
    }}

    [data-testid="stFileUploaderDropzoneInstructions"] span {{
        color: var(--di-text) !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }}

    [data-testid="stFileUploaderDropzoneInstructions"] small {{
        color: var(--di-faint) !important;
    }}

    [data-testid="stFileUploaderDropzone"] button {{
        background: transparent !important;
        color: var(--di-accent) !important;
        border: 1.5px solid var(--di-accent) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }}

    [data-testid="stFileUploaderDropzone"] button p {{ color: var(--di-accent) !important; }}

    [data-testid="stFileUploaderDropzone"] button:hover {{
        background: var(--di-accent) !important;
    }}

    [data-testid="stFileUploaderDropzone"] button:hover p {{ color: #ffffff !important; }}

    /* Compact variant used in the sidebar's "add more" panel */
    .di-dropzone-compact [data-testid="stFileUploaderDropzone"] {{
        padding: 20px 12px !important;
        border-radius: 14px !important;
    }}

    .di-dropzone-compact [data-testid="stFileUploaderDropzoneInstructions"]::before {{
        width: 40px;
        height: 40px;
        margin-bottom: 10px;
    }}

    [data-testid="stFileUploaderFile"] {{
        background: var(--di-card);
        border: 1px solid var(--di-border);
        border-radius: 10px;
    }}

    /* ------------------------------------------------------------------ */
    /* Chat messages                                                       */
    /* ------------------------------------------------------------------ */

    div[data-testid="stChatMessage"] {{
        background: var(--di-card);
        border: 1px solid var(--di-border);
        border-radius: 16px;
        padding: 6px 10px;
        margin-bottom: 4px;
        box-shadow: var(--di-shadow);
    }}

    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {{
        color: var(--di-text);
        line-height: 1.55;
    }}

    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] code {{
        font-family: var(--di-font-mono);
        background: var(--di-input);
        border: 1px solid var(--di-border);
        color: var(--di-accent-light);
        padding: 1px 6px;
        border-radius: 5px;
        font-size: 0.85em;
    }}

    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] pre {{
        background: var(--di-input) !important;
        border: 1px solid var(--di-border);
        border-radius: 10px;
    }}

    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] pre code {{
        background: transparent;
        border: none;
        color: var(--di-text);
    }}

    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
        background: linear-gradient(180deg, {ACCENT}14, var(--di-card));
        border-color: {ACCENT}44;
    }}

    .di-meta {{
        font-family: var(--di-font-mono);
        font-size: 0.7rem;
        color: var(--di-faint);
        margin: 2px 4px 4px 4px;
    }}

    /* ------------------------------------------------------------------ */
    /* Buttons (main content area)                                         */
    /* ------------------------------------------------------------------ */

    .stButton > button {{
        border-radius: 10px;
        border: 1px solid var(--di-border);
        background: var(--di-card);
        color: var(--di-text) !important;
        font-family: var(--di-font-body);
        transition: border-color 0.15s ease, color 0.15s ease, background 0.15s ease;
    }}

    .stButton > button p {{ color: var(--di-text) !important; }}

    .stButton > button:hover {{
        border-color: var(--di-accent);
        color: var(--di-accent-light) !important;
        background: var(--di-card-hover);
    }}

    .stButton > button:hover p {{ color: var(--di-accent-light) !important; }}

    .di-primary-btn button,
    .st-key-di_hero_process_btn .stButton > button {{
        background: linear-gradient(135deg, {ACCENT}, {ACCENT_DARK});
        color: #ffffff !important;
        border: none;
        font-weight: 600;
    }}

    .di-primary-btn button p,
    .st-key-di_hero_process_btn .stButton > button p {{ color: #ffffff !important; }}

    .di-primary-btn button:hover,
    .st-key-di_hero_process_btn .stButton > button:hover {{
        opacity: 0.92;
        color: #ffffff !important;
    }}

    /* Send button -- round blue "launch" action, scoped via st.container(key=...) */
    .st-key-di_send_btn .stButton > button {{
        background: linear-gradient(135deg, {BEACON}, {ACCENT_DARK});
        border: none;
        color: #ffffff !important;
        font-weight: 700;
        font-size: 1.1rem;
        height: 100%;
        min-height: 60px;
        border-radius: 14px;
        box-shadow: 0 4px 16px {BEACON}55;
    }}
    .st-key-di_send_btn .stButton > button p {{
        color: #ffffff !important;
    }}
    .st-key-di_send_btn .stButton > button:hover {{
        opacity: 0.9;
        transform: translateY(-1px);
    }}

    [data-testid="stDownloadButton"] button {{
        background: var(--di-card);
        color: var(--di-text) !important;
        border: 1px solid var(--di-border);
    }}
    [data-testid="stDownloadButton"] button p {{ color: var(--di-text) !important; }}
    [data-testid="stDownloadButton"] button:hover {{
        border-color: var(--di-accent);
        color: var(--di-accent-light) !important;
    }}
    [data-testid="stDownloadButton"] button:hover p {{ color: var(--di-accent-light) !important; }}

    /* ------------------------------------------------------------------ */
    /* Text inputs / text areas                                            */
    /* ------------------------------------------------------------------ */

    textarea, .stTextInput input, [data-testid="stTextAreaRootElement"] {{
        background: var(--di-input) !important;
        color: var(--di-text) !important;
        border-radius: 12px !important;
        border: 1px solid var(--di-border) !important;
        font-family: var(--di-font-body) !important;
    }}

    textarea::placeholder, .stTextInput input::placeholder {{
        color: var(--di-faint) !important;
        opacity: 1;
    }}

    textarea:focus, .stTextInput input:focus {{
        border-color: var(--di-accent) !important;
        box-shadow: 0 0 0 3px {ACCENT}2a !important;
    }}

    .stApp [data-testid="stHorizontalBlock"] [data-testid="stTextArea"] textarea {{
        border-radius: 16px !important;
        padding: 14px 16px !important;
    }}

    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] .stTextInput input {{
        background: var(--di-sidebar-soft) !important;
        color: var(--di-sidebar-text) !important;
        border: 1px solid var(--di-sidebar-border) !important;
    }}
    [data-testid="stSidebar"] textarea::placeholder,
    [data-testid="stSidebar"] .stTextInput input::placeholder {{
        color: var(--di-sidebar-faint) !important;
    }}

    [data-testid="stSidebar"] .stNumberInput input,
    [data-testid="stSidebar"] .stDateInput input {{
        background: var(--di-sidebar-soft) !important;
        color: var(--di-sidebar-text) !important;
        border: 1px solid var(--di-sidebar-border) !important;
    }}

    /* ------------------------------------------------------------------ */
    /* Labels                                                               */
    /* ------------------------------------------------------------------ */

    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label {{
        color: var(--di-muted) !important;
        font-size: 0.85rem;
    }}

    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] label {{
        color: var(--di-sidebar-muted) !important;
    }}

    /* ------------------------------------------------------------------ */
    /* Selectbox -- including the dropdown, which renders in a portal      */
    /* ------------------------------------------------------------------ */

    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
        background: var(--di-sidebar-soft) !important;
        color: var(--di-sidebar-text) !important;
        border-color: var(--di-sidebar-border) !important;
        border-radius: 12px !important;
    }}

    [data-testid="stSelectbox"] div[data-baseweb="select"] {{
        background: var(--di-sidebar-soft) !important;
    }}

    [data-testid="stSelectbox"] div[data-baseweb="select"] span {{
        color: var(--di-sidebar-text) !important;
    }}

    [data-testid="stSelectbox"] div[data-baseweb="select"] * {{
        color: var(--di-sidebar-text) !important;
    }}

    [data-testid="stSelectbox"] [role="button"] {{
        color: var(--di-sidebar-text) !important;
    }}

    div[data-baseweb="popover"] ul[role="listbox"],
    div[data-baseweb="popover"] div[data-baseweb="menu"] {{
        background: var(--di-sidebar-soft) !important;
        border: 1px solid var(--di-sidebar-border) !important;
        border-radius: 12px !important;
        box-shadow: var(--di-shadow) !important;
    }}

    div[data-baseweb="popover"] li[role="option"] {{
        color: var(--di-sidebar-text) !important;
        font-family: var(--di-font-body);
        background: var(--di-sidebar-soft) !important;
    }}

    div[data-baseweb="popover"] li[role="option"]:hover,
    div[data-baseweb="popover"] li[aria-selected="true"] {{
        background: var(--di-sidebar-active) !important;
        color: var(--di-accent-light) !important;
    }}

    [data-testid="stMultiSelect"] span[data-baseweb="tag"] {{
        background: var(--di-accent-dark) !important;
        color: #ffffff !important;
    }}

    /* ------------------------------------------------------------------ */
    /* Expander                                                             */
    /* ------------------------------------------------------------------ */

    [data-testid="stExpander"] {{
        border: 1px solid var(--di-border) !important;
        border-radius: 12px !important;
        overflow: hidden;
        background: var(--di-card);
    }}

    [data-testid="stSidebar"] [data-testid="stExpander"] {{
        background: var(--di-sidebar-soft);
        border: 1px solid var(--di-sidebar-border) !important;
    }}

    [data-testid="stExpander"] summary {{
        color: var(--di-text) !important;
        font-weight: 500;
    }}

    [data-testid="stSidebar"] [data-testid="stExpander"] summary {{
        color: var(--di-sidebar-text) !important;
        cursor: pointer;
        transition: color 0.15s ease;
    }}

    [data-testid="stExpander"] summary:hover {{
        color: var(--di-accent-light) !important;
    }}

    [data-testid="stExpander"] svg {{
        fill: var(--di-muted) !important;
    }}

    [data-testid="stExpanderDetails"] {{
        border-top: 1px solid var(--di-border-soft);
        background: var(--di-card);
        color: var(--di-text);
    }}

    [data-testid="stSidebar"] [data-testid="stExpanderDetails"] {{
        background: var(--di-sidebar-soft);
        border-top-color: var(--di-sidebar-border);
        color: var(--di-sidebar-text) !important;
    }}

    [data-testid="stSidebar"] [data-testid="stExpanderDetails"] * {{
        color: var(--di-sidebar-text) !important;
    }}

    /* ------------------------------------------------------------------ */
    /* Toggle                                                               */
    /* ------------------------------------------------------------------ */

    [data-testid="stToggle"] label p {{
        color: var(--di-sidebar-text) !important;
    }}

    [data-testid="stToggle"] [role="switch"][aria-checked="true"] {{
        background: var(--di-accent) !important;
    }}

    /* ------------------------------------------------------------------ */
    /* Metrics                                                              */
    /* ------------------------------------------------------------------ */

    [data-testid="stMetric"] {{
        background: var(--di-sidebar-soft);
        border: 1px solid var(--di-sidebar-border);
        border-radius: 12px;
        padding: 10px 14px;
    }}

    [data-testid="stMetricLabel"] p {{
        color: var(--di-sidebar-faint) !important;
        font-family: var(--di-font-mono);
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}

    [data-testid="stMetricValue"] {{
        color: var(--di-accent-light) !important;
        font-family: var(--di-font-display);
    }}

    /* Sidebar column alignment for document library */
    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {{
        display: flex;
        align-items: center;
        flex-wrap: nowrap;
    }}

    [data-testid="stSidebar"] [data-testid="stColumn"] {{
        display: flex;
        align-items: center;
        min-height: 44px;
    }}

    [data-testid="stSidebar"] [data-testid="stColumn"] [data-testid="stVerticalBlock"] {{
        display: flex;
        align-items: center;
        justify-content: center;
    }}

    /* ------------------------------------------------------------------ */
    /* Alerts (warning / success / error)                                   */
    /* ------------------------------------------------------------------ */

    [data-testid="stAlert"] {{
        border-radius: 12px;
        font-family: var(--di-font-body);
    }}

    [data-testid="stAlert"] p {{
        color: inherit !important;
    }}

    /* ------------------------------------------------------------------ */
    /* Spinner                                                              */
    /* ------------------------------------------------------------------ */

    [data-testid="stSpinner"] p {{
        color: var(--di-muted) !important;
        font-family: var(--di-font-body);
    }}

    /* ------------------------------------------------------------------ */
    /* Misc                                                                 */
    /* ------------------------------------------------------------------ */

    hr {{ border-color: var(--di-border); }}
    [data-testid="stSidebar"] hr {{ border-color: var(--di-sidebar-border); }}

    [data-testid="stCaptionContainer"] {{ color: var(--di-faint) !important; }}
</style>
"""