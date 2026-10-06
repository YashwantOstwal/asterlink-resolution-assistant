from typesafe_sdk import Choice


SYSTEM_PROMPT_KB = """
You are a high-precision extraction engine for AsterLink knowledge base articles.

Your only job is to extract information already present in the provided knowledge
base text. Never invent, infer, classify, paraphrase, or extrapolate information.

### INPUT FORMAT

You will receive unstructured or semi-structured text containing one knowledge base
article, including information such as KB ID, Title, Description, Resolution Steps,
and Page Number.

### OUTPUT FORMAT

Return ONLY a serialized JSON object as plain text.

Your entire response must begin with { and end with }.

DO NOT wrap the response in Markdown code fences.
DO NOT use ```json.
DO NOT use ```.
DO NOT include explanations, labels, introductory text, or trailing text.

The response must contain exactly these six keys:

{
  "title": "<string>",
  "description": "<string>",
  "resolution_steps": ["<string>", "<string>", "..."],
  "kb_id": "<string>",
  "is_valid_kb": "<boolean>",
  "page_number": "<number>"
}

The response must be directly parseable with:

json.loads(response.content)

### STRICT EXTRACTION RULES

1. title:
   - Extract the primary title explicitly present in the input.
   - Preserve the original wording where possible.
   - Return "" if missing.

2. description:
   - Extract the description or issue context explicitly present in the input.
   - Return "" if missing.

3. resolution_steps:
   - Return a JSON array of strings.
   - Extract every individual resolution step in its original order.
   - Remove only numbering, bullets, or leading list symbols.
   - Do not combine distinct steps.
   - Return [] if no resolution steps are present.

4. kb_id:
   - Extract the KB identifier explicitly present in the input.
   - Return "" if missing.

5. is_valid_kb:
   - Return true if and only if kb_id is non-empty.
   - Otherwise return false.

6. page_number:
   - Extract the page number explicitly provided in the input.
   - Return it as a JSON number, not a string.
   - Do not infer or calculate a page number.
   - Return 0 if the page number is missing or unavailable.

7. Never invent, infer, classify, paraphrase, or extrapolate information.

### FINAL RESPONSE CONSTRAINT

Your first character MUST be {
Your last character MUST be }

Return nothing outside those characters.
"""
SYSTEM_PROMPT_GENERATION = """
You are AsterLink's customer-support resolution assistant.

Given a CURRENT CUSTOMER COMPLAINT, retrieved KNOWLEDGE BASE ARTICLES, and
SIMILAR RESOLVED TICKETS, generate a concise and actionable resolution using
only the supplied evidence.

### SOURCE ROLES

- Knowledge-base articles are AsterLink's approved procedures and are the
  primary source of truth.
- Historical tickets are supporting examples of how similar complaints were
  resolved. Similarity does not prove the current complaint has the same cause.
- If a ticket conflicts with a knowledge-base article, follow the knowledge base.

### GROUNDING RULES

1. Use only information from the supplied KB articles and historical tickets.
2. Never invent procedures, diagnoses, root causes, customer/account/network
   facts, resolution steps, source IDs, file names, or page numbers.
3. Use historical resolution steps only when applicable to the current complaint.
4. When multiple KBs apply, combine their procedures into one logical resolution.
   Remove duplicate or redundant steps rather than concatenating sources blindly.
5. Prefer steps directly relevant to the customer's symptoms.
6. Ignore retrieved sources that do not materially contribute to the resolution.
7. Cite only sources actually used. Preserve their IDs, file names, and page
   numbers exactly as provided.
8. If evidence is insufficient, return only supported steps. If no actionable
   step is supported, return an empty resolution_steps array.
9. If sources conflict, prefer the applicable KB procedure and omit unresolved
   or unsupported actions.

Order the resolution logically when applicable:
validate the problem -> diagnose -> remediate -> escalate if required ->
verify resolution.

### OUTPUT

Return ONLY a serialized/stringified JSON object with exactly these keys:

{
  "resolution_steps": [
    "<actionable resolution step>"
  ],
}

Rules:
- page_number must be a JSON number.
- Do not include duplicate sources.
- Do not add additional keys.
- Do not include source identifiers inside resolution_steps.
- The output must be directly parseable by json.loads(response.content).
- Return no Markdown, code fences, explanations, or text outside the JSON.
- The first character must be { and the last character must be }.
"""



SYSTEM_PROMPT_TICKET = """
You are a high-precision JSON extraction engine for AsterLink resolved support tickets.

Extract only information explicitly present in the ticket.
Do not infer, classify, summarize, reinterpret, normalize, or invent values.

Return ONLY a serialized/stringified JSON object as plain response text.

The entire response must be directly parseable using:

json.loads(response.content)

Do NOT return:
- Markdown
- ```json code fences
- Explanations
- A Python dictionary
- A JSON-encoded string containing escaped JSON
- Any extra keys

Use exactly this JSON structure:

{
  "ticket_id": <string>,
  "complaint": <string>,
  "product": <string>,
  "category": <string>,
  "severity": <string>,
  "customer_sentiment": <string>,
  "resolution_steps": "<string>, "..."],
  "is_valid_ticket": <boolean>
}

Rules:

- Extract all values exactly as present in the ticket.
- Return "" for any missing string field.
- Return [] if resolution_steps are missing.
- Each resolution step must be a separate string.
- Preserve the original order of resolution steps.
- Never use null.
- Do not infer missing values.
- is_valid_ticket must be true only when ticket_id is non-empty.

Return only the serialized/stringified JSON object.
"""




CLASSIFICATION_QUESTIONS = {
    "product": Choice(
        instructions="Identify the primary product or service affected. Base the decision on the main customer problem rather than secondary symptoms or technologies mentioned.",
        criteria={
            "Home Internet": (
                "The customer's fixed home internet connection. Includes fiber, DSL, "
                "5G/LTE fixed wireless, gateway/WAN service, broadband activation, outages, "
                "repeated drops, slow speeds, latency, or other access-network problems. "
                "Use when the broadband connection itself is affected. Do not use when "
                "broadband works normally and only local Wi-Fi is affected."
            ),

            "Home Wi-Fi": (
                "The local wireless network inside the premises. Use when the underlying "
                "internet works but Wi-Fi has weak signal, dead zones, authentication or "
                "association failures, interference, mesh problems, or extender problems."
            ),

            "Mobile Service": (
                "General cellular service on a mobile line. Includes no service, emergency "
                "calls only, network registration, mobile data, cellular performance, "
                "calling, messaging, or geographic coverage when a more specific feature "
                "is not the main problem. A SIM/eSIM may be mentioned without making it "
                "the primary product."
            ),

            "SIM/eSIM": (
                "Physical SIM or eSIM lifecycle and provisioning. Use when SIM/eSIM "
                "activation, replacement, transfer, profile download, QR/profile state, "
                "device association, or SIM registration itself is the primary failure. "
                "Do not use merely because a SIM is mentioned in a broader mobile-service issue."
            ),

            "Mobile Hotspot": (
                "Mobile hotspot or tethering. Use when the phone's own mobile data works "
                "but connected devices cannot access the internet correctly, hotspot "
                "entitlement is missing, or tethering performance/configuration is the "
                "main problem."
            ),

            "Voicemail": (
                "Voicemail or Visual Voicemail. Use when mailbox access, callers leaving "
                "messages, Visual Voicemail loading, voicemail notifications, or voicemail "
                "provisioning is the main problem."
            ),

            "Wi-Fi Calling": (
                "The mobile Wi-Fi Calling feature. Use when normal Wi-Fi works but "
                "Wi-Fi Calling specifically cannot activate, register, remain enabled, "
                "or reliably place or receive calls."
            ),
        },
    ),

    "category": Choice(
        instructions="Identify the primary customer-visible failure mode. Classify what is mainly failing rather than the underlying technical cause.",
        criteria={
            "Connectivity": (
                "The service cannot establish or maintain a usable connection. Includes "
                "complete internet outage, repeated disconnects, WAN/access-link failure, "
                "no mobile service, emergency calls only, searching for network, cellular "
                "registration failure, or area-wide service loss. If the service remains "
                "connected but is only slow, use Performance."
            ),

            "Performance": (
                "The service remains available but performs significantly worse than "
                "expected. Includes slow throughput, buffering, high latency, high ping, "
                "jitter, packet loss, congestion, or slow mobile data."
            ),

            "Wi-Fi": (
                "The failure is specifically in the local Wi-Fi network while the underlying "
                "internet connection remains available. Includes weak signal, dead zones, "
                "wireless-only disconnects, device association/authentication failures, "
                "interference, mesh pairing, or extender/backhaul problems."
            ),

            "Activation & Provisioning": (
                "A service, gateway, plan, feature, or network configuration has not been "
                "correctly activated, provisioned, associated, or configured. Includes "
                "pending activation, account/network provisioning mismatch, missing service "
                "profiles, failed gateway provisioning, or configuration preventing service."
            ),

            "Device & SIM": (
                "The SIM/eSIM lifecycle or its association with the device is itself the "
                "primary failure. Includes replacement SIM registration, SIM moves, eSIM "
                "transfers, stuck eSIM activation, stale profiles, consumed QR codes, or "
                "incorrect SIM/device mapping. General network registration failure belongs "
                "under Connectivity unless the SIM/eSIM lifecycle is clearly central."
            ),

            "Messaging": (
                "The primary customer-visible failure concerns SMS, MMS, group messaging, "
                "or RCS. Use Messaging even when APN, data, or provisioning settings are "
                "the technical cause."
            ),

            "Voice": (
                "The primary customer-visible failure concerns calling or a voice-related "
                "service. Includes incoming/outgoing call failure, dropped calls, robotic "
                "audio, one-way audio, silence during calls, voicemail, or Wi-Fi Calling."
            ),

            "Tethering": (
                "The primary failure concerns mobile hotspot or tethering. Includes "
                "hotspot-connected devices having no internet, poor tethered performance, "
                "hotspot entitlement problems, or tethering configuration limits."
            ),
        },
    ),

    "customer_sentiment": Choice(
        instructions="Identify the emotional tone explicitly expressed by the customer. Do not infer sentiment from technical severity.",
        criteria={
            "neutral": (
                "The complaint is factual and emotionally unmarked. There is no clear "
                "expression of worry, fear, irritation, or dissatisfaction."
            ),

            "worried": (
                "The customer expresses worry, fear, anxiety, apprehension, or concern "
                "about the problem or its consequences. The dominant emotion is concern "
                "about what is happening or what may happen."
            ),

            "frustrated": (
                "The customer expresses irritation, annoyance, dissatisfaction, or "
                "impatience. Common signals include repeated failures, failed troubleshooting, "
                "the issue happening again, or explicit frustration with the service."
            ),
        },
    ),

    "severity": Choice(
        instructions="Determine severity from operational impact only. Do not use the customer's emotional tone when assigning severity.",
        criteria={
            "high": (
                "The primary service is unavailable, nearly unavailable, or repeatedly "
                "fails enough to prevent normal use. Includes total mobile-service loss, "
                "complete internet outage, fiber/DSL access failure, persistent network "
                "registration failure, major calling failure, severe recurring disconnects, "
                "or major area-wide service loss."
            ),

            "medium": (
                "There is meaningful degradation or partial loss, but important service "
                "functionality remains available. Includes slow internet, latency or "
                "buffering, mobile-data failure while calls/texts work, messaging problems, "
                "hotspot problems, Wi-Fi Calling failure, partially usable connectivity, "
                "or incomplete activation without severe established-service loss."
            ),

            "low": (
                "The problem is narrow or localized while the main service remains usable. "
                "Includes one-device Wi-Fi problems, minor/localized Wi-Fi coverage issues, "
                "mesh/extender-only problems, or voicemail-only failures while core calling "
                "and data remain available."
            ),
        },
    ),
}