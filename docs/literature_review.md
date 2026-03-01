# Literature and Background Review

## 1. Traditional Approaches in Real Estate Valuation
Historically, real estate valuation models have relied heavily on traditional statistical techniques, primarily Hedonic Pricing Models (HPM). These models deconstruct a property's value into its constituent characteristics (e.g., number of bedrooms, square footage, neighborhood). While interpretable, HPMs and conventional linear regressions struggle to capture complex, non-linear interactions between features. 

More recently, tree-based ensemble methods such as Random Forest and XGBoost have become the industry standard for tabular data in real estate due to their robustness to outliers and ability to model non-linear boundaries. However, these models are strictly limited to structured, tabular datasets and ignore unstructured data like property descriptions and images.

## 2. Text Mining and Sentiment in Real Estate
Property listings often contain rich, unstructured text (amenities, location highlights). Natural Language Processing (NLP) techniques, particularly transformer-based models like BERT (Bidirectional Encoder Representations from Transformers), have shown superior capabilities in extracting contextual embeddings from text. For the Indian context, cultural factors such as *Vastu Shastra* (the traditional Indian system of architecture) play a vital subjective role in the desirability and premium pricing of a property. Models utilizing NLP to parse listing descriptions for keywords related to Vastu compliance represent a significant, yet under-explored, gap in automated valuation models.

## 3. Computer Vision in Property Assessment
Visual appeal is intuitively one of the most critical factors driving real estate sales. Traditional models fail to account for the physical condition, lighting, or luxury aesthetics of a property. Recent advancements in Computer Vision, notably Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs), allow for the quantification of visual data. ViTs, in particular, treat image patches as sequences, efficiently learning global context and outperforming CNNs on various image classification tasks. Fine-tuning a ViT to extract a "Visual Premium Score" provides a novel mechanism to ingest aesthetic value into an algorithmic pricing engine.

## 4. Multimodal Fusion Learning
No single data modality (tabular, textual, or visual) can perfectly capture a property's true market value and Return on Investment (ROI) potential. Multimodal Fusion Learning aims to combine embeddings from diverse data sources into a single predictive framework. Recent literature demonstrates that cross-attention mechanisms—where representations from one modality (e.g., images) attend to another (e.g., text or tabular features)—yield highly robust predictive models.

## 5. Conclusion & Architecture Choice
Our project, **BharatPropAdvisor**, builds upon these state-of-the-art developments by implementing a novel, end-to-end multimodal fusion architecture. By utilizing an FT-Transformer for geospatial and tabular data, an NLP branch for Vastu keyword extraction, and a Vision Transformer (ViT-base) for visual scoring, our custom cross-attention layer aims to synthesize these disparate inputs to offer holistic, high-accuracy ROI predictions in the Indian real estate market.
