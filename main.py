import json

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma

from dotenv import load_dotenv

load_dotenv()


knowledge_base = [
    {
        "kb_id": "KB-NET-001",
        "title": "Complete Home Internet Outage",
        "description": "Use this procedure when the customer's home internet is completely unavailable rather than merely slow or unstable. Typical reports include no device being able to get online, websites and applications failing to load, a router or gateway remaining powered while internet access is unavailable, or Wi-Fi remaining visible while all connected devices report no internet connection. The problem may affect fiber, DSL, or fixed-wireless home internet. When multiple devices fail at the same time, the likely cause is beyond a single phone or laptop and may involve the WAN session, gateway provisioning, access line, customer-premises equipment, an upstream network failure, or an area outage. This article should be distinguished from intermittent connectivity, where service repeatedly returns, and from weak Wi-Fi coverage, where only particular devices or locations are affected.",
        "resolution_steps": [
            "Confirm whether all connected devices are affected and whether the local gateway or router remains reachable.",
            "Check gateway power, physical cabling, WAN or session state, service provisioning, and any current area outage or maintenance event.",
            "Restore or reprovision the broadband session when the WAN state or service profile is inconsistent.",
            "Correct a supported gateway or physical connection issue when it can be safely resolved remotely.",
            "Escalate unresolved access-line or upstream network failures with equipment state and timestamps attached.",
            "Verify internet access on a known-good device before closing the case."
        ]
    },

    {
        "kb_id": "KB-NET-002",
        "title": "Intermittent Broadband and Recurring Internet Drops",
        "description": "Use this procedure when internet access repeatedly disconnects and returns instead of remaining completely offline. Customers may report short drops several times per day, failures around the same time each evening, frozen work calls, disconnected games, or internet access disappearing briefly while the Wi-Fi network remains visible. A router restart may restore service temporarily before the problem returns. If both wired and wireless devices lose connectivity together, investigate the WAN session, access line, optical or radio connection, gateway stability, or upstream congestion. If Ethernet remains stable while only Wi-Fi devices fail, investigate local wireless coverage instead.",
        "resolution_steps": [
            "Determine whether the interruption affects Wi-Fi only or the full internet connection, including wired devices where available.",
            "Record the frequency, duration, and approximate timestamps of the connection drops.",
            "Review gateway, WAN session, access-line, optical, or radio-link health around the affected periods.",
            "Correct supported gateway firmware, configuration, or session instability when identified.",
            "Refresh or reprovision an unstable WAN session where appropriate.",
            "Escalate recurring upstream instability or congestion with timestamps and diagnostic evidence.",
            "Confirm stability during a representative observation period after remediation."
        ]
    },

    {
        "kb_id": "KB-NET-003",
        "title": "Slow Home Internet Throughput",
        "description": "Use this procedure when home internet remains available but delivers substantially lower throughput than expected. Customers may report buffering video, slow downloads, long page-load times, poor file-transfer performance, or speed-test results far below the subscribed plan. Performance may also become worse during evening peak periods. Possible causes include weak Wi-Fi, device limitations, an incorrect speed profile, access-line errors, gateway load, radio conditions, or network congestion. For fixed broadband, a wired or near-gateway test is important because poor Wi-Fi performance does not necessarily indicate a problem with the broadband service itself. If throughput remains normal but interactive applications suffer from delay, use the latency procedure instead.",
        "resolution_steps": [
            "Compare measured throughput with the subscribed plan and the active service profile.",
            "Test on more than one capable device and, where possible, perform a wired or near-gateway speed test.",
            "Check local Wi-Fi conditions, gateway load, access-line quality, radio quality, and network congestion indicators.",
            "Correct a confirmed local device or gateway bottleneck.",
            "Reapply the correct speed or service profile when provisioning is inconsistent with the active plan.",
            "Escalate persistent access-network or capacity degradation with repeatable test evidence.",
            "Repeat the controlled performance test after remediation to verify improvement."
        ]
    },

    {
        "kb_id": "KB-NET-004",
        "title": "High Latency, Jitter, Packet Loss, or Ping Spikes",
        "description": "Use this procedure when bandwidth appears acceptable but interactive applications perform poorly because of excessive delay or unstable packet delivery. Customers may report high ping, gaming lag, jitter, packet loss, stuttering voice or video calls, remote-desktop freezes, delayed responses, or applications that feel slow despite normal download and upload speeds. A connection may carry large amounts of data while still performing badly for real-time traffic. Causes can include local wireless interference, unstable access links, congested network paths, packet loss, routing instability, or peak-hour congestion.",
        "resolution_steps": [
            "Measure latency, jitter, and packet loss separately from download and upload throughput.",
            "Compare results at different times and, for fixed broadband, compare wired and Wi-Fi performance.",
            "Record affected applications, destinations, timestamps, and any repeatable packet-loss or latency pattern.",
            "Correct local wireless interference when the issue is isolated to Wi-Fi.",
            "Correct supported access-link instability when packet loss originates on the customer connection.",
            "Escalate repeatable upstream latency, congestion, packet loss, or routing faults with measurements attached.",
            "Repeat latency and packet-loss testing after remediation."
        ]
    },

    {
        "kb_id": "KB-NET-005",
        "title": "Weak Wi-Fi Signal, Dead Zones, or Wireless Interference",
        "description": "Use this procedure when the underlying broadband service is working but Wi-Fi performance is poor in particular rooms, floors, or areas of the premises. Customers may report strong service next to the router but weak signal in bedrooms, upper floors, garages, or distant offices. Devices may disconnect when moved away from the gateway, or streaming may buffer only in certain parts of the home. A key indicator is that Ethernet or devices close to the gateway remain stable while distant wireless devices experience weak signal or poor performance. Common causes include gateway placement, walls and floors, neighboring wireless networks, interference, inappropriate frequency-band selection, or insufficient wireless coverage.",
        "resolution_steps": [
            "Confirm that the underlying broadband service is stable using Ethernet or a near-gateway test.",
            "Compare Wi-Fi signal and performance across affected and unaffected locations.",
            "Review gateway placement, distance, physical obstructions, band selection, and wireless channel congestion.",
            "Adjust supported Wi-Fi settings or gateway placement where appropriate.",
            "Use an approved extender or mesh solution when one gateway cannot adequately cover the premises.",
            "Verify wireless performance in the previously affected locations after changes."
        ]
    },

    {
        "kb_id": "KB-NET-006",
        "title": "Unable to Join the Home Wi-Fi Network",
        "description": "Use this procedure when a device can see the customer's home Wi-Fi network but cannot successfully connect. Typical reports include repeated password prompts, authentication errors, connection failures, or a new phone, laptop, television, game console, or smart-home device that cannot join the network. Possible causes include incorrect credentials, an outdated saved network profile, unsupported security settings, band compatibility, access controls, or gateway Wi-Fi configuration. If devices successfully join Wi-Fi but cannot reach the internet, use the complete internet outage procedure instead.",
        "resolution_steps": [
            "Confirm the correct Wi-Fi network name and current password.",
            "Check whether other devices can connect successfully.",
            "Remove an outdated saved network profile from the affected device and reconnect where appropriate.",
            "Review supported security mode, band compatibility, device access controls, and gateway Wi-Fi status.",
            "Restore approved Wi-Fi settings when gateway configuration is incorrect.",
            "Escalate suspected gateway hardware or firmware faults when multiple compatible devices cannot connect."
        ]
    },

    {
        "kb_id": "KB-NET-007",
        "title": "Router or Gateway Configuration Lost After Reset, Replacement, or Update",
        "description": "Use this procedure when internet connectivity stops or customer settings disappear after a router, modem, or gateway has been reset, replaced, reconfigured, or updated. Customers may report that Wi-Fi remains available after a factory reset but internet access no longer works, a replacement gateway cannot establish service, or network settings disappeared following an update. A healthy physical broadband connection does not guarantee internet access if the gateway has lost the correct WAN configuration, service profile, credentials, VLAN parameters, or provider provisioning. Establish that the underlying access connection is healthy before changing gateway configuration.",
        "resolution_steps": [
            "Confirm whether the problem began after a reset, replacement, configuration change, or firmware update.",
            "Verify that the underlying physical access link is healthy.",
            "Compare the gateway's current WAN and service configuration with the approved account profile.",
            "Check whether required credentials, VLAN parameters, or managed provisioning are missing.",
            "Restore or reprovision the approved WAN and service configuration.",
            "Restore supported customer Wi-Fi settings after the service path is working.",
            "Escalate when the gateway cannot accept the correct profile or appears defective.",
            "Verify stable internet access after the configuration is restored."
        ]
    },

    {
        "kb_id": "KB-NET-008",
        "title": "Fiber Optical Signal Loss, Red LOS, or Unstable PON",
        "description": "Use this procedure for fiber customers reporting a red or blinking LOS indicator, unstable PON status, optical warning, or complete service loss associated with the optical network terminal. A persistent LOS condition generally indicates that the ONT is not receiving the expected optical signal. The issue may begin after construction, maintenance, movement of the fiber cable, damage to the customer-side connection, or an upstream optical-network failure. Restarting the Wi-Fi router usually does not repair a genuine optical fault because the problem occurs before the router's internet session.",
        "resolution_steps": [
            "Confirm whether the ONT LOS indicator is red or blinking and whether the state is persistent.",
            "Check PON and other optical-status indicators.",
            "Ask whether the issue followed construction, maintenance, or movement of the fiber cable.",
            "Review provider-side ONT registration and optical status where available.",
            "Check for nearby fiber incidents or access-network outages.",
            "Restore the optical or access-network path remotely when supported.",
            "Arrange field or fiber-access repair when optical signal cannot be restored remotely.",
            "Verify normal ONT optical status and internet connectivity after repair."
        ]
    },

    {
        "kb_id": "KB-NET-009",
        "title": "DSL Line Not Synchronizing or Broadband Light Not Stable",
        "description": "Use this procedure when a DSL customer reports that the modem cannot establish line synchronization, the DSL or broadband light remains off or flashing, or the internet service cannot establish even after restarting the modem. Possible causes include poor line quality, incorrect filtering, loose cabling, inside-wiring faults, service-profile problems, or an access-network failure. This differs from a Wi-Fi problem because the modem itself cannot establish the underlying DSL connection.",
        "resolution_steps": [
            "Confirm the DSL or broadband indicator state and whether the modem detects the access line.",
            "Check the wall connection, approved filters, telephone equipment, and modem cabling.",
            "Review line-quality and synchronization status where available.",
            "Reprovision the DSL service profile when the physical line is healthy but session establishment remains inconsistent.",
            "Escalate persistent line-synchronization failures for access-line investigation.",
            "Verify stable DSL synchronization after remediation."
        ]
    },

    {
        "kb_id": "KB-NET-010",
        "title": "5G or LTE Home Internet Weak Signal and Gateway Placement",
        "description": "Use this procedure when fixed-wireless home internet is unavailable, unstable, or slow because the home gateway has weak cellular signal or poor placement. Customers may report a red or amber gateway status, low network signal, service that improves when the gateway is moved near a window, repeated connection drops, or major performance differences depending on gateway location. Fixed-wireless gateways depend on both the cellular access connection and the customer's local Wi-Fi network. Weak cellular access signal should therefore be distinguished from local Wi-Fi coverage problems.",
        "resolution_steps": [
            "Check the gateway's cellular signal and network-status indicators.",
            "Confirm that the gateway is positioned in an approved location with minimal obstruction and adequate network signal.",
            "Compare service stability and performance after relocating the gateway within supported placement guidance.",
            "Separate weak cellular access signal from local Wi-Fi coverage problems.",
            "Refresh network registration when supported after finalizing gateway placement.",
            "Escalate persistent weak-signal or coverage failures when no supported placement provides reliable service.",
            "Verify stable service after the final gateway placement."
        ]
    }
]

ticket_archive = [
    {
        "ticket_id": "TKT-0001",
        "complaint": "My fiber box and router are both on, but since this morning every phone and laptop in the house says there is no internet.",
        "product": "Fiber Home Internet",
        "category": "Connectivity",
        "intent": "Report complete internet outage",
        "severity": "high",
        "customer_sentiment": "concerned",
        "resolution_steps": [
            "Confirmed that the outage affected all connected devices and that the local gateway was still reachable.",
            "Checked gateway power, physical connections, WAN session state, service provisioning, and current outage status.",
            "Found the broadband session in an inconsistent state and reprovisioned the service before re-establishing the WAN connection.",
            "Verified restored internet access on a known-good device before closing the ticket."
        ]
    },

    {
        "ticket_id": "TKT-0004",
        "complaint": "My broadband keeps dropping during work calls several times a day. Restarting the router helps for a short while, but the problem always comes back and the calls become choppy before the connection drops.",
        "product": "Home Broadband",
        "category": "Intermittent Connectivity",
        "intent": "Report recurring broadband drops",
        "severity": "high",
        "customer_sentiment": "frustrated",
        "resolution_steps": [
            "Confirmed that the interruptions affected the full internet connection rather than only the customer's work laptop.",
            "Collected timestamps for the broadband drops and the periods of degraded call quality.",
            "Reviewed gateway and access-link diagnostics around the affected periods and found repeated WAN instability.",
            "Measured packet loss and latency during the degraded-call periods to distinguish the real-time quality problem from insufficient bandwidth.",
            "Applied the supported gateway and session correction for the recurring connection instability.",
            "Escalated the remaining upstream instability and packet-loss pattern with timestamps and measurements attached."
        ]
    },

    {
        "ticket_id": "TKT-0005",
        "complaint": "I pay for 500 Mbps fiber but even on Ethernet I am only getting around 70 Mbps today. Wi-Fi is not the issue because the wired test is slow too.",
        "product": "Fiber Home Internet",
        "category": "Performance",
        "intent": "Report slow internet speed",
        "severity": "medium",
        "customer_sentiment": "frustrated",
        "resolution_steps": [
            "Compared the measured throughput with the customer's subscribed plan and active service profile.",
            "Repeated the speed test on another capable device using a wired connection.",
            "Checked gateway load, access-line quality, and network congestion indicators.",
            "Found that the provisioned speed profile did not match the active service tier and reapplied the correct profile.",
            "Repeated the controlled wired speed test and confirmed that throughput returned to the expected range."
        ]
    }
]

new_complaints = [
    {
        "complaint": "Wi-Fi still shows up normally, but none of us can open anything online. I restarted the modem and later reset it, but the connection is still completely dead.",
        "expected_relevant_kb_ids": [
            "KB-NET-001",
            "KB-NET-007"
        ],
        "expected_similar_ticket_ids": [
            "TKT-0001"
        ]
    },
    {
        "complaint": "The internet cuts out for a minute or two almost every night around 8 PM. Wi-Fi stays connected and then everything starts working again by itself.",
        "expected_relevant_kb_ids": [
            "KB-NET-002"
        ],
        "expected_similar_ticket_ids": [
            "TKT-0004"
        ]
    },
    {
        "complaint": "My home internet still works, but downloads have become painfully slow this week and video keeps buffering in the evenings. It is noticeably worse in the upstairs bedroom than next to the router.",
        "expected_relevant_kb_ids": [
            "KB-NET-003",
            "KB-NET-005"
        ],
        "expected_similar_ticket_ids": [
            "TKT-0005"
        ]
    }
]

def calculate_recall_at_k(retreived, relevant, k):
    # relevant in top-k retrieved / total relevant 

    return len(set(retreived[:k]) & set(relevant)) / len(relevant)

def calculate_precision_at_k(retrieved, relevant, k):
    # relevant in top-k retrieved /  retrieved
    retrieved_at_k = retrieved[:k]
    if not retrieved_at_k:
        return 0.0
    return len(set(retrieved_at_k) & set(relevant)) / len(retrieved_at_k)


def main():
    ticket_archive_chunks = []
    for ticket in ticket_archive:
        ticket_archive_chunk = Document(page_content = ticket.get("complaint",""),metadata = {
            "info": json.dumps({
                "ticket_id": ticket.get("ticket_id"),
                "resolution_steps": ticket.get("resolution_steps"),
            })
        })
        ticket_archive_chunks.append(ticket_archive_chunk)
    

    kb_chunks = []
    for kb in knowledge_base:
        kb_chunk = Document(page_content = f"{kb.get("title")}. {kb.get("description")}",metadata = {
            "info": json.dumps({
                "kb_id": kb.get("kb_id"),
                "resolution_steps": kb.get("resolution_steps"),
            })
        })
        kb_chunks.append(kb_chunk)

    embedding_model = OpenAIEmbeddings(model = "text-embedding-3-small")

    knowledge_base_directory = "db/knowledge_base"
    # knowledge_base_corpus = Chroma.from_documents(
    #     documents=kb_chunks,
    #     embedding=embedding_model,
    #     persist_directory=knowledge_base_directory, 
    #     collection_metadata={"hnsw:space": "cosine"}
    # )
    knowledge_base_corpus = Chroma(
        embedding_function=embedding_model,
        persist_directory=knowledge_base_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )

    tickets_archive_directory = "db/tickets_archive"
    # tickets_corpus = Chroma.from_documents(
    #     documents=ticket_archive_chunks,
    #     embedding=embedding_model,
    #     persist_directory = tickets_archive_directory, 
    #     collection_metadata={"hnsw:space": "cosine"}
    # )
    tickets_corpus = Chroma(
        embedding_function=embedding_model,
        persist_directory=tickets_archive_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )

    knowledge_base_retriever = knowledge_base_corpus.as_retriever(
        search_type = "similarity_score_threshold",
        search_kwargs={
            "k": 2,
            "score_threshold": 0.5 
        }
    )

    
    ticket_archive_retriever = tickets_corpus.as_retriever(
        search_type = "similarity_score_threshold",
        search_kwargs={
            "k": 2,
            "score_threshold": 0.4 
        }
    )

    llm = ChatOpenAI(model = "gpt-4o")
    for new_complaint in new_complaints[:1]: 
        retrieved_kb_chunks = knowledge_base_retriever.invoke(new_complaint.get("complaint"))
        retrieved_kb_ids = []

        for retrieved_kb_chunk in retrieved_kb_chunks:
            info = json.loads(retrieved_kb_chunk.metadata.get("info"))
            retrieved_kb_ids.append(info.get("kb_id"))

        recall_at_2_for_kb_chunks = calculate_recall_at_k(retrieved_kb_ids,new_complaint.get("expected_relevant_kb_ids"),2)
        precision_at_2_for_kb_chunks = calculate_precision_at_k(retrieved_kb_ids,new_complaint.get("expected_relevant_kb_ids"),2)
        # print(recall_at_2_for_kb_chunks,precision_at_2_for_kb_chunks,retrieved_kb_ids,"\n")

        retrieved_ticket_archive_chunks = ticket_archive_retriever.invoke(new_complaint.get("complaint"))
        retrieved_ticket_ids = []
        for retrieved_kb_chunk in retrieved_ticket_archive_chunks:
            info = json.loads(retrieved_kb_chunk.metadata.get("info"))
            retrieved_ticket_ids.append(info.get("ticket_id"))

        recall_at_2_for_ticket_archive_chunks = calculate_recall_at_k(retrieved_ticket_ids,new_complaint.get("expected_similar_ticket_ids"),2)
        precision_at_2_for_ticket_archive_chunks = calculate_precision_at_k(retrieved_ticket_ids,new_complaint.get("expected_similar_ticket_ids"),2)
        # print(recall_at_2_for_ticket_archive_chunks,precision_at_2_for_ticket_archive_chunks,retriretrieved_ticket_idseved_kb_ids,"\n")

        system_message = """
You are a telecom customer-support resolution assistant.

Your job is to help a customer support agent resolve a customer's complaint using the retrieved company knowledge-base articles and similar previously resolved support tickets provided in the user message.

SOURCE ROLES:
- Knowledge-base articles contain the company's approved troubleshooting and resolution procedures.
- Similar resolved tickets contain historical examples of how related customer complaints were diagnosed and resolved.
- Use the knowledge base as the primary source of truth.
- Use historical tickets as supporting evidence for choosing and adapting an appropriate resolution path.

GROUNDING RULES:
1. Base the resolution only on information contained in the provided knowledge-base articles and similar resolved tickets.
2. Do not invent troubleshooting procedures, diagnostic findings, account information, network conditions, customer actions, or resolution steps that are not supported by the provided context.
3. Do not assume that a historical ticket's diagnosis is also true for the current customer. Historical tickets are examples, not evidence that the current complaint has the same root cause.
4. When multiple retrieved knowledge-base articles are relevant, combine their procedures into one coherent resolution plan.
5. Prefer knowledge-base procedures when a historical ticket conflicts with a knowledge-base article.
6. Avoid duplicate or redundant steps when combining information from multiple sources.
7. Present the steps in a logical troubleshooting order: validate the problem, diagnose likely causes, apply supported remediation, escalate when required, and verify resolution.
8. Do not mention retrieved sources that are unrelated to the final resolution.
9. Cite only the knowledge-base articles and historical tickets that materially contributed to the generated resolution.
10. Never fabricate a KB ID or ticket ID.

RESPONSE FORMAT:

## Recommended Resolution

1. <first actionable resolution step>
2. <second step>
3. <third step>
...

## Sources Used

**Knowledge Base:** <comma-separated KB IDs actually used, or "None">

**Similar Tickets:** <comma-separated TKT IDs actually used, or "None">

Keep the response concise, actionable, and suitable for a telecom support agent.
"""
        human_message = f"""
CURRENT CUSTOMER COMPLAINT:
{new_complaint.get("complaint")}

RELEVANT KNOWLEDGE BASE ARTICLES:
The following are retrieved knowledge-base articles. Each entry contains
article content followed by its metadata.
"""

        for kb_chunk in retrieved_kb_chunks:
            human_message += f"""
---
CONTENT:
{kb_chunk.page_content}

METADATA:
{kb_chunk.metadata}
"""

        human_message += """
        SIMILAR RESOLVED TICKETS:
        The following are retrieved historical customer complaints. Each entry
        contains the complaint followed by metadata describing how the historical
        case was resolved.
        """

        for ticket in retrieved_ticket_archive_chunks:
            human_message += f"""
---
COMPLAINT:
{ticket.page_content}

METADATA:
{ticket.metadata}
"""

        human_message += """
        Using only the evidence above, produce the recommended resolution for the
        current customer complaint according to the required response format.
        """


        messages = [SystemMessage(content = system_message),HumanMessage(content = human_message)]
        response = llm.invoke(messages)

        print(response.content) 
        

if __name__ == "__main__": 
    main()

