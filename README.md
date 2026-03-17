# Data

The dataset used in this project (`trans50m_2k18.xlsx`) contains all publicly announced M&A transactions exceeding USD 50M since 2018. It was provided by Albert School as part of the Quant Trading Club curriculum and is not included in this repository.

## Schema (38,655 transactions × 14 columns)

| Column | Type | Description |
|---|---|---|
| `All Transactions Announced Date` | datetime | Date the transaction was announced |
| `All Transactions Announced Date (Including Bids and Letters of Intent)` | datetime | Announcement date including preliminary bids |
| `Target/Issuer` | string | Name of the target company |
| `Exchange:Ticker` | string | Exchange and ticker symbol of the target |
| `Transaction Types` | string | All rows are "Merger/Acquisition" |
| `Transaction Status` | string | Closed, Announced, Withdrawn, etc. |
| `Total Transaction Value ($USDmm, Historical rate)` | float | Deal value in USD millions |
| `Buyers/Investors` | string | Acquiring entity name(s) |
| `Sellers` | string | Selling entity name(s) |
| `CIQ Transaction ID` | string | Unique transaction identifier (S&P Capital IQ) |
| `Transaction Comments` | string | Free-text deal description |
| `Country/Region of Incorporation [Target/Issuer]` | string | Target company country |
| `Country/Region of Incorporation [Sellers]` | string | Seller country (may contain multiple, semicolon-separated) |
| `Target Security Types` | string | Type of security involved |

## Reproducing

To run the notebook, place a file matching this schema in `data/trans50m_2k18.xlsx`. Equivalent datasets can be exported from S&P Capital IQ or similar M&A databases.
