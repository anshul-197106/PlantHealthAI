def simulated_spraying_decision(predicted_class, confidence, severity_result):
    """Generate a display-only sprinkler status. No hardware or chemical dose."""
    label = predicted_class.lower()
    if confidence < 0.60:
        return {"level": "Inspection required", "action": "Retake the image or ask an agricultural expert to inspect the plant.", "reason": "The screening result is uncertain, so the simulated sprinkler remains off."}
    if "healthy" in label:
        return {"level": "Standby", "action": "Continue monitoring the plant.", "reason": "The image classifier predicted a healthy class."}
    if severity_result.get("status") not in {"validated", "manual_demo"}:
        return {"level": "Review required", "action": "Confirm the diagnosis and severity before treatment.", "reason": "The image-based severity estimate is not validated, so no automatic spray response is suggested."}
    if severity_result.get("status") == "manual_demo":
        value = float(severity_result.get("value", 0))
        if value < 10:
            return {"level": "Demo: low", "action": "Simulated sprinkler standby; inspect and monitor.", "reason": "Illustrative demo threshold only; not an agronomic recommendation."}
        if value < 40:
            return {"level": "Demo: medium", "action": "Simulated sprinkler request; human review required.", "reason": "Illustrative demo threshold only; no real pump or valve is connected."}
        return {"level": "Demo: high", "action": "Simulated sprinkler request; human review required.", "reason": "Illustrative demo threshold only; no real pump or valve is connected."}
    return {"level": "Review required", "action": "Use crop-specific, label-approved guidance after expert confirmation.", "reason": "Validated rules and safe equipment integration have not been configured."}
