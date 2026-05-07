# Design System: Pure No-Nonsense (v7.0)
## Scientific Lab Terminal Aesthetic

This design system is optimized for high-density biometric data analysis. It prioritizes functional clarity, industrial precision, and algorithmic transparency over commercial decoration.

---

## 1. Core Principles
- **Linear Stream**: Information flows in a vertical, full-width stream. No side-by-side card clusters.
- **Border-First**: Structure is defined by subtle borders (`1px obsidian/10`) rather than shadows or depth.
- **Monochromatic Base**: Heavy use of Bone White and Obsidian, with Deep Royal Blue as the single semantic anchor.
- **Data Over Design**: Raw bitstreams and diagnostic metrics are integrated as first-class visual elements.

---

## 2. Color Tokens
| Token | Hex | Usage |
| :--- | :--- | :--- |
| **Background (Bone)** | `#fcfcfc` | Main workspace background. |
| **Primary (Royal)** | `#004085` | Primary actions, Hamming scores, active states. |
| **Text (Obsidian)** | `#2d3436` | Primary headings and body text. |
| **Muted (Slate)** | `#636e72` | Labels, metadata, secondary diagnostic text. |
| **Success (Teal)** | `#006266` | Verification passed, secure states. |
| **Error (Crimson)** | `#b71540` | Verification failed, pipeline interruptions. |
| **Surface (Bone/50)** | `rgba(252, 252, 252, 0.5)` | Section headers and subtle row highlights. |

---

## 3. Typography & Typescale
We use the **Overpass** family for its industrial, clean, and highly readable characteristics.

| Role | Size | Font | Weight | Color | Intent |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Hero Display** | `2xl` | Mono | Bold | Royal | High-signal metrics (Hamming). |
| **Primary Title** | `xl` | Sans | Bold | Obsidian | Main identification targets. |
| **Secondary Header** | `text-xs` | Sans | Bold | Obsidian | Section headers, Sidebar links. |
| **Standard Body** | `text-sm` | Sans | Regular | Obsidian | Inputs, Tabular data. |
| **Technical Utility** | `[10px]` | Mono | Bold | Slate | Metadata, Confidences, IDs. |
| **Micro Data** | `[10px]` | Mono | Regular | Slate/60 | Raw bitstreams, binary segments. |

---

## 4. Aesthetic Strategies for Legibility
- **The 10px Floor**: No text shall be smaller than `10px`. Hierarchy is achieved through weight and color, not extreme miniaturization.
- **Weight as Contrast**: Technical utility text (IDs, confidences) uses `Bold` weights to remain "scannable" at small sizes.
- **Color De-emphasis**: Secondary labels and micro-data use `Slate Muted` (`#636e72`) with optional opacity (`0.6`) to push them into the background without sacrificing font size.
- **The Royal Anchor**: The Deep Royal Blue is reserved strictly for high-value data and primary state triggers, acting as a visual magnet.

---

## 5. Layout & Spacing
- **Header**: Top-anchored, sticky (`top-0`), high-density navigation.
- **Main Container**: Full-width viewport focus (`w-full`).
- **Responsive Strategy**: 
  - Desktop: Horizontal navigation, multi-column diagnostic grids.
  - Mobile: Vertical stack for all elements, full-width data rows.
- **Global Spacing Cap**: Padding and Margin are limited to a maximum of `4` (`1rem`).
- **Section Gaps**: Standardized at `gap-4`.
- **Internal Padding**: `p-2` to `p-4` for high-density containers.
- **Iconography**: Fixed size of `20px` (`w-5 h-5`) for action buttons. **Header navigation is text-only**.
- **Borders**: `1px solid rgba(45, 52, 54, 0.1)`.

---

## 5. Components
- **Input Pipeline**: Full-width dashed upload area (`border-2 border-dashed`).
- **Action Buttons**: Right-aligned, anchored at the bottom of their respective sections.
- **Data Tables**: High-density, no icons, consistent row-hover states (`hover:bg-bone/30`).
- **Diagnostic Grid**: 4-column split using internal vertical dividers.

---

## 6. Iconography
- **Source**: RemixIcon (`ri-` prefix).
- **Style**: Line style, subtle coloring (`slate-muted`), used only for functional indicators (Scan, Database, Enroll).
