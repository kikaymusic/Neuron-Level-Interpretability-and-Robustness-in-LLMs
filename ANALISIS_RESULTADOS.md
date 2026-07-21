# Análisis completo de resultados — Revisión SYNAPSE (DGX Spark)

_Generado: 2026-07-15. Datos: `spark_out/data/{BigBird,Longformer,goemotions}` y `spark_out2/data/{BERT,GPT2,DistilBERT}`._
_Baselines del paper tomados de `revisionSYNAPSE/document_pdf-2.md` (Table 4/5/6)._

## Decisiones de configuración
- **Silencing global se deja a 1 pasada (50 muestras), SIN promediar semillas** (decisión del usuario, 2026-07-15). Detección / bit-flip / random-control sí van con 5 semillas.
- Métrica: nuestro `detection_and_f1_malware.csv` reporta **macro-F1**; el paper Table 4 reporta **weighted-F1**. Datos: subconjunto balanceado 500/clase.

---

## Veredicto rápido
Ha ido bien y es **netamente continuista** con el paper. Ningún resultado a nivel de azar (bug viejo resuelto), el orden de los modelos coincide, y las tres tesis se reproducen:
1. **Jerarquía de vulnerabilidad escalonada:** ruido/fault-sneaking = leve → silencing = fuerte tras umbral → **bit-flip = catastrófico**.
2. **Los modelos de alta capacidad (BigBird/Longformer/GPT-2) colapsan más** bajo silenciamiento que BERT/DistilBERT.
3. **Las neuronas del probe son causalmente importantes** (control aleatorio) y **coinciden con atribución por gradiente** (conductance/IG).

Valor añadido por la revisión: GPT-2 (decoder), 5 semillas + held-out, conductance, contraste cross-domain con GoEmotions.

---

## 1) Detección / baseline (5 semillas)

| Modelo | ROC-AUC | macro-F1 (nuestro) | Paper Table 4 (weighted-F1) |
|---|---|---|---|
| BigBird | 0.976 | 0.879 | 0.8306 |
| Longformer | 0.973 | 0.875 | 0.8516 |
| GPT-2 | 0.885 | 0.625 | — (nuevo) |
| DistilBERT | 0.851 | 0.576 | 0.6081 |
| BERT | 0.830 | 0.601 | 0.6834 |

Detectores sólidos (ROC-AUC ≥ 0.83). Mismo orden que el paper. GPT-2 en el medio (decoder). Diferencias absolutas = macro-vs-weighted F1 + datos balanceados 500/clase.

## 2) Silenciamiento global (weighted-F1, degradación)

| Modelo | baseline | @20% | @50% | @95% | ∆ aprox |
|---|---|---|---|---|---|
| BERT | 0.506 | 0.493 | 0.414 | 0.314 | −38% |
| DistilBERT | 0.482 | 0.471 | 0.407 | 0.278 | −42% |
| BigBird | 0.735 | 0.397 | 0.159 | 0.173 | −76% |
| Longformer | 0.716 | 0.618 | 0.571 | 0.173 | −76% |
| GPT-2 | 0.625 | 0.546 | 0.141 | 0.069 | −89% |

Patrón "meseta y caída" idéntico al paper (Table 5): estable hasta ~10-20%, luego colapso. Los modelos grandes se desploman mucho más (−76/−89%) que BERT/DistilBERT (−38/−42%). GPT-2 se comporta como los grandes.
Nota: curvas de 1 pasada → ruidosas/no-monótonas (BigBird 12.5%=0.46, 15%=0.67). Decisión: se acepta así, sin semillas.

## 3) Bit-flip (BFA) — hallazgo más fuerte

| Modelo | baseline | @2% | @5% | suelo |
|---|---|---|---|---|
| BERT | 0.601 | 0.197 | 0.092 | 0.071 |
| DistilBERT | 0.576 | 0.578 | 0.371 | 0.071 |
| BigBird | 0.879 | 0.157 | 0.067 | 0.065 |
| Longformer | 0.875 | 0.179 | 0.067 | 0.065 |
| GPT-2 | 0.625 | 0.217 | 0.121 | 0.072 |
| **GoEmotions** | **0.951** | **0.921** | **0.941** | **0.931** |

Devastador: con solo 2-5% de flips todos los detectores de malware caen al suelo (~0.07). Narrativa Terminal Brain Damage / BFA. DistilBERT aguanta el 2% (el más pequeño) y luego cae.
**Contraste nuevo y potente:** GoEmotions (NLP general) apenas se inmuta (−1 a −3%) → los detectores de malware son **específicamente frágiles** al bit-flip, no es artefacto genérico. Refuerza la tesis de seguridad.

## 4) Control aleatorio (validación causal)

Caída con top-k vs neuronas aleatorias (drop en F1):

| Modelo | top-drop (alto %) | random-drop | z-score máx |
|---|---|---|---|
| BigBird | 0.80 | 0.06–0.34 | ~23 |
| Longformer | 0.82 | 0.02–0.44 | ~22 |
| GoEmotions | 0.81 | 0.03 | ~30 |
| DistilBERT | 0.43 | ~0.01 | ~23 |
| BERT | 0.27 | ~0.03 | ~17 |

Las neuronas del probe importan mucho más que las aleatorias (p_empírico llega al mínimo 0.048), sobre todo BigBird/Longformer/GoEmotions. Validación causal: el ranking no es artefacto. En BERT/DistilBERT el efecto es más débil (modelos más distribuidos).

## 5) Acuerdo de atribución (probe vs conductance / act×grad)

| Modelo | Spearman (probe vs conductance) |
|---|---|
| Longformer | 0.701 |
| DistilBERT | 0.629 |
| BERT | 0.515 |
| BigBird | 0.464 |
| **GPT-2** | **0.146** (outlier) |

Correlaciones positivas y significativas en encoders (0.46–0.70) → el ranking del probe se alinea con la atribución por gradiente. **GPT-2 outlier (0.15):** decoder + mean-pool; hay que discutirlo.

## 6) Ruido gaussiano y fault-sneaking (robustez suave)

| Modelo | Noise 10% σ=0.1 (acc) | Fault-sneak 10% (wF1) |
|---|---|---|
| BERT | 0.56 | 0.50 |
| DistilBERT | 0.54 | 0.46 |
| BigBird | 0.74 | 0.77 |
| Longformer | 0.80 | 0.82 |

Caídas modestas → bajo ruido/fault-sneaking los modelos aguantan (los grandes casi intactos). Coincide con el paper. Completa la jerarquía: ruido ≪ silencing ≪ bit-flip.

## 7) Coste computacional

Extracción de activaciones: BERT 10s, DistilBERT 7s, **BigBird 209s, Longformer 244s** (~25× por 4096 tokens); GPU ~1GB vs ~2.1GB. GPT-2 total 844s. Coherente con longitud de contexto.

---

## Cosas a vigilar antes de escribir
1. ~~Silencing/noise a 1 pasada~~ → **aceptado sin semillas** (decisión usuario). Declarar como config, no como limitación bloqueante.
2. **Métrica y datos:** dejar claro macro-F1 (nuestro) vs weighted-F1 (paper) y subconjunto balanceado 500/clase.
3. **GPT-2 atribución baja (0.15)** → párrafo de discusión (decoder/mean-pool).
4. **R1.4/R4.5 pendiente:** reconciliar baseline BigBird (Table 4 = 0.8306 vs Table 5 = 0.785).
5. **CORREGIDO 2026-07-17 — la versión anterior de esta nota era ERRÓNEA.** Decía que bajo silenciamiento
   los modelos colapsan hacia "Normal" y dejan pasar malware. Eso solo es cierto en el **régimen bajo**
   (2.5-5%: clase 2=Normal con recall 1.0, precisión 0.44, sobre-predicción leve de Normal).
   **En el colapso real (≥50%) ocurre lo contrario:** todas las clases caen a recall 0.0 salvo la clase 4
   (TheTick, que es MALWARE), que se queda con recall 0.88-1.0 y precisión 0.32. Es decir, el modelo marca
   **todo como malicioso**. Consistente con `detection_and_f1_malware.csv`: BigBird @50% → FAR=1.000,
   MAR=0.000, TPR@1%FPR=0.329.
   → Argumento correcto: el fallo es **ruidoso**, no silencioso. Satura de falsas alarmas y, en un pipeline
   de respuesta automática, bloquea tráfico legítimo en masa. NO es "deja pasar malware".
   → Matiz honesto: la clase 4 es la mayoritaria del subconjunto (support 17/50), así que el atractor del
   colapso es un artefacto de la distribución, no una preferencia aprendida. No sobrevender.

## GoEmotions (cross-domain) — extra
- Baseline 0.951 (28-clase emoción).
- Control aleatorio: top-k causalmente importante (z hasta 30 @65%), misma "meseta y caída" → el patrón generaliza fuera de malware.
- CLS vs mean-pool agreement Spearman 0.419 (moderado).
- Bit-flip: casi intacto (contraste con malware, ver §3).
