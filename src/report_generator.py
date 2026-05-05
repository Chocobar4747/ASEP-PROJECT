"""
LLM Report Generator (Groq)
==============================
Generates human-readable analysis reports from dye concentration predictions.
Uses Groq's free API with template-based fallback.
"""

import os
import logging
from typing import Dict, Optional


def _get_concentration_level(ppm: float, max_ppm: float = 200.0) -> str:
    ratio = ppm / max_ppm
    if ratio < 0.15:
        return "very low"
    elif ratio < 0.35:
        return "low"
    elif ratio < 0.55:
        return "moderate"
    elif ratio < 0.75:
        return "high"
    else:
        return "very high"


def _get_safety_note(ppm: float, dye_type: str) -> str:
    dye_notes = {
        "methylene_blue": {"name": "Methylene Blue", "safe_limit": 50,
            "warning": "concentrations above 50 ppm may indicate excessive dye usage"},
        "congo_red": {"name": "Congo Red", "safe_limit": 30,
            "warning": "known to be potentially carcinogenic; minimize in effluent"},
        "crystal_violet": {"name": "Crystal Violet", "safe_limit": 40,
            "warning": "can be toxic to aquatic organisms at elevated concentrations"},
    }
    info = dye_notes.get(dye_type, {"name": dye_type.replace("_", " ").title(),
                                      "safe_limit": 50, "warning": "elevated concentrations should be monitored"})
    if ppm > info["safe_limit"]:
        return f"⚠️ Note: {info['warning']}."
    return "✅ The detected concentration is within typical acceptable limits."


def generate_report_template(
    ppm: float, dye_type: str = "unknown",
    rgb_values: Optional[Dict[str, float]] = None,
    model_name: str = "ML Model",
    confidence_metrics: Optional[Dict[str, float]] = None,
) -> str:
    level = _get_concentration_level(ppm)
    safety = _get_safety_note(ppm, dye_type)
    dye_name = dye_type.replace("_", " ").title()

    report = f"""
╔══════════════════════════════════════════════════════════╗
║           DYE CONCENTRATION ANALYSIS REPORT             ║
╚══════════════════════════════════════════════════════════╝

📋 SAMPLE ANALYSIS SUMMARY
  Dye Type          : {dye_name}
  Predicted Conc.   : {ppm:.2f} ppm
  Concentration     : {level.upper()}
  Model             : {model_name}
"""
    if rgb_values:
        report += f"""
🎨 COLOR ANALYSIS
  Red (mean)   : {rgb_values.get('r_mean', 0):.1f}
  Green (mean) : {rgb_values.get('g_mean', 0):.1f}
  Blue (mean)  : {rgb_values.get('b_mean', 0):.1f}
"""
    if confidence_metrics:
        report += f"""
📊 MODEL CONFIDENCE
  R² Score : {confidence_metrics.get('r2', 0):.4f}
  RMSE     : {confidence_metrics.get('rmse', 0):.3f} ppm
"""
    report += f"""
🔬 INTERPRETATION
  The sample shows a {level} concentration of {dye_name}
  at approximately {ppm:.2f} ppm.
  {safety}
"""
    return report.strip()


def generate_report_groq(
    ppm: float, dye_type: str = "unknown",
    rgb_values: Optional[Dict[str, float]] = None,
    model_name: str = "ML Model",
    confidence_metrics: Optional[Dict[str, float]] = None,
    api_key: Optional[str] = None,
) -> str:
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        logging.warning("GROQ_API_KEY not set. Using template.")
        return generate_report_template(ppm, dye_type, rgb_values, model_name, confidence_metrics)

    try:
        from groq import Groq
        client = Groq(api_key=key)
        dye_name = dye_type.replace("_", " ").title()
        level = _get_concentration_level(ppm)

        rgb_info = ""
        if rgb_values:
            rgb_info = (f"Mean RGB: R={rgb_values.get('r_mean',0):.1f}, "
                        f"G={rgb_values.get('g_mean',0):.1f}, B={rgb_values.get('b_mean',0):.1f}.")

        prompt = (f"Write a concise professional dye concentration analysis report (150 words).\n"
                  f"Dye: {dye_name}, Concentration: {ppm:.2f} ppm ({level}), Model: {model_name}. {rgb_info}\n"
                  f"Include: summary, interpretation, safety notes for this dye.")

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a chemistry lab analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3, max_tokens=400,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        logging.warning(f"Groq error: {e}. Falling back to template.")
        return generate_report_template(ppm, dye_type, rgb_values, model_name, confidence_metrics)
