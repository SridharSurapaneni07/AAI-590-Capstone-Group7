# Introduction

## Problem Statement
Millions of Indian real estate buyers and investors face a complex decision-making process: "Given a fixed budget and time horizon, what type of property, in which city and locality, will yield the maximum total returns (capital appreciation + rental yield)?" Current solutions available in the market (e.g., brokers, online portals like 99acres or MagicBricks, and basic price predictors) suffer from several critical shortcomings:
- They are often biased towards promoted listings.
- They lack objective, data-driven Return on Investment (ROI) predictions.
- They ignore essential cultural factors that significantly impact property value and desirability in India, such as Vastu Shastra compliance.
- They do not utilize visual analysis of actual property photos to quantify the premium appeal of a property.

## Proposed Solution: BharatPropAdvisor
To address these gaps, we propose **BharatPropAdvisor**, a pan-India multimodal Transformer-based recommender system. Our system enables users to input their budget and constraints (and optionally upload 3–5 photos of a specific property) to instantly receive:
1. Top 5 ranked property recommendations across India.
2. Predicted ROI % for 3-year and 5-year horizons.
3. Vastu compliance score based on structural facing and keywords, along with actionable tips.
4. A Visual Premium Score (0–100) derived from analyzing uploaded photos, complete with explainable AI (Grad-CAM heatmaps).
5. Comprehensive risk level assessments, mispricing alerts (identifying "undervalued gems"), an interactive map, and automated one-click PDF reports.

## Project Scope
This project fulfills the requirements of the AAI590 Capstone (USD MS-AAI). By combining a novel multimodal fusion architecture (Vision + Text + Tabular/Geospatial Transformers) wrapped in a live, user-facing Streamlit application, BharatPropAdvisor delivers high real-world impact in India’s $700B+ real estate market. It addresses a widely untouched intersection of ROI prediction, cultural text extraction (Vastu), and visual evaluation.
