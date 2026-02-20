# Spatiotemporal-Wildfire-Spread-Prediction-Using-ConvLSTM-and-Satellite-Data

Wildfire frequency and intensity have escalated globally, threatening ecosystems, economies, and human safety.
We present an eight‑page comprehensive report on a ConvLSTM‑based framework that predicts next‑day
wildfire spread exclusively from environmental satellite data—removing the common dependency on prior fire
masks. Using MODIS burned‑area products, MODIS vegetation and surface‑temperature indices, and ERA5
meteorological re‑analysis, we formulate wildfire forecasting as an end‑to‑end spatiotemporal segmentation task.
Our model, trained on the 2021 Manavgat (Turkey) mega‑fire and validated on geographically held‑out regions,
achieves an Intersection‑over‑Union (IoU) of 0.22 and a recall of 0.91. Although precision remains low (0.003)
due to extreme class imbalance, the system exceeds the widely accepted 0.20 IoU threshold for operational
"rapid‑response" usability. We discuss data preprocessing, training strategy, results, limitations, and future
work—such as multi‑scale attention and synthetic augmentation to improve minority‑class representation.

**Keywords: Wildfire spread prediction; ConvLSTM; Spatiotemporal deep learning; Remote sensing; MODIS
MCD64A1; ERA5 meteorology; Manavgat 2021; Next-day forecasting
**
