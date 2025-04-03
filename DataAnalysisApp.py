import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Page setup
st.set_page_config(page_title="Aktien Analyse Tool", layout="wide")
st.title("📈 Professioneller Aktienanalysator")


# Cache functions
@st.cache_data
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="5y")
        earnings = stock.earnings
        quarterly_earnings = stock.quarterly_earnings
        recommendations = stock.recommendations
        return info, hist, earnings, quarterly_earnings, recommendations
    except Exception as e:
        st.error(f"Fehler beim Abrufen der Daten: {str(e)}")
        return None, None, None, None, None


@st.cache_data
def monte_carlo_simulation(hist_data, days=30, simulations=1000):
    log_returns = np.log(hist_data['Close'] / hist_data['Close'].shift(1)).dropna()
    mean = log_returns.mean()
    std_dev = log_returns.std()

    last_price = hist_data['Close'].iloc[-1]
    simulations_result = np.zeros((days, simulations))

    for i in range(simulations):
        prices = [last_price]
        for _ in range(days):
            prices.append(prices[-1] * np.exp(np.random.normal(mean, std_dev)))
        simulations_result[:, i] = prices[1:]

    return simulations_result


# UI Components
ticker = st.sidebar.text_input("Aktiensymbol eingeben (z.B. AAPL, MSFT):", "AAPL").upper()

if st.sidebar.button("Analyse starten", type="primary"):
    if not ticker:
        st.warning("Bitte geben Sie ein Aktiensymbol ein")
        st.stop()

    with st.spinner("Daten werden geladen..."):
        info, hist_data, earnings, quarterly_earnings, recommendations = get_stock_data(ticker)

        if info is None or hist_data is None:
            st.error("Konnte keine Daten für dieses Symbol abrufen")
            st.stop()

        # Unternehmensbeschreibung
        if 'longBusinessSummary' in info:
            st.subheader(f"Unternehmensprofil: {info.get('shortName', ticker)}")
            st.write(info['longBusinessSummary'])

        # Sektoren- und Brancheninfo
        if 'sector' in info and 'industry' in info:
            st.subheader("📌 Sektor & Branche")
            cols = st.columns(2)
            cols[0].metric("Sektor", info['sector'])
            cols[1].metric("Branche", info['industry'])

        # Analystenbewertungen
        st.subheader("📊 Analystenbewertungen")
        cols = st.columns(4)

        if 'recommendationMean' in info:
            cols[0].metric("Durchschnittliche Bewertung",
                           f"{info['recommendationMean']:.1f}",
                           "1=Strong Buy, 5=Strong Sell")

        if 'targetMeanPrice' in info:
            current = info.get('currentPrice', 0)
            target = info['targetMeanPrice']
            diff = (target - current) / current * 100
            cols[1].metric("Mittleres Kursziel",
                           f"${target:.2f}",
                           f"{'+' if diff > 0 else ''}{diff:.1f}% vs. aktuell")

        if 'numberOfAnalystOpinions' in info:
            cols[2].metric("Analystenmeinungen",
                           info['numberOfAnalystOpinions'])

        if recommendations is not None and not recommendations.empty:
            try:
                # Check for different possible column names
                rec_columns = recommendations.columns
                if 'To Grade' in rec_columns:
                    latest_rec = recommendations.iloc[0]['To Grade']
                elif 'grade' in rec_columns:
                    latest_rec = recommendations.iloc[0]['grade']
                elif 'action' in rec_columns:
                    latest_rec = recommendations.iloc[0]['action']
                else:
                    # Try to get the first non-date column if standard columns not found
                    non_date_cols = [col for col in rec_columns if
                                     not pd.api.types.is_datetime64_any_dtype(recommendations[col])]
                    if non_date_cols:
                        latest_rec = recommendations.iloc[0][non_date_cols[0]]
                    else:
                        latest_rec = "Keine Daten"
            except Exception as e:
                st.warning(f"Konnte Empfehlungsdaten nicht lesen: {str(e)}")
                latest_rec = "Fehler"

            cols[3].metric("Letzte Empfehlung", latest_rec)
        else:
            cols[3].metric("Letzte Empfehlung", "Nicht verfügbar")

        # Earnings Analyse
        st.subheader("💰 Earnings Analyse")

        if earnings is not None and not earnings.empty:
            # Aktuelle Schätzungen
            if 'earningsQuarterlyGrowth' in info:
                growth = info['earningsQuarterlyGrowth']
                st.metric("Erwartetes Quartalswachstum (YoY)",
                          f"{growth * 100:.2f}%" if growth else "N/A")

            # Vorherige Earnings vs. Schätzungen
            if quarterly_earnings is not None and not quarterly_earnings.empty:
                latest_quarter = quarterly_earnings.iloc[0]
                if 'Revenue' in latest_quarter and 'Earnings' in latest_quarter:
                    cols = st.columns(2)

                    # Revenue Analyse
                    if 'revenueEstimate' in info:
                        revenue_diff = latest_quarter['Revenue'] - info['revenueEstimate']
                        cols[0].metric("Umsatz (letztes Quartal)",
                                       f"${latest_quarter['Revenue'] / 1e9:.2f} Mrd.",
                                       f"{'Überboten' if revenue_diff > 0 else 'Unterboten'} um ${abs(revenue_diff) / 1e6:.2f} Mio.")

                    # Earnings Analyse
                    if 'earningsEstimate' in info:
                        earnings_diff = latest_quarter['Earnings'] - info['earningsEstimate']
                        cols[1].metric("Gewinn (letztes Quartal)",
                                       f"${latest_quarter['Earnings']:.2f} pro Aktie",
                                       f"{'Überboten' if earnings_diff > 0 else 'Unterboten'} um ${abs(earnings_diff):.2f}")

            # Earnings Chart
            st.markdown("### Historische Earnings")
            if 'Earnings' in earnings:
                fig_earn, ax_earn = plt.subplots(figsize=(12, 4))
                earnings['Earnings'].plot(kind='bar', ax=ax_earn, color='green')
                ax_earn.set_title('Jährliche Earnings')
                ax_earn.set_ylabel('Gewinn pro Aktie ($)')
                st.pyplot(fig_earn)

        # Bilanzkennzahlen
        st.subheader("🏦 Bilanzkennzahlen")
        cols = st.columns(4)

        if 'totalDebt' in info and 'totalCash' in info:
            net_debt = (info['totalDebt'] - info['totalCash']) / 1e9
            cols[0].metric("Netto-Verschuldung",
                           f"${net_debt:.2f} Mrd.",
                           "Total Debt - Cash")

        if 'returnOnEquity' in info:
            cols[1].metric("Eigenkapitalrendite (ROE)",
                           f"{info['returnOnEquity'] * 100:.1f}%")

        if 'returnOnAssets' in info:
            cols[2].metric("Kapitalrendite (ROA)",
                           f"{info['returnOnAssets'] * 100:.1f}%")

        if 'operatingMargins' in info:
            cols[3].metric("Operative Marge",
                           f"{info['operatingMargins'] * 100:.1f}%")

        # Wichtige Kennzahlen
        st.subheader("📊 Wichtige Kennzahlen")
        cols = st.columns(4)
        metrics = [
            ("Aktueller Preis", info.get('currentPrice', 'N/A'), "$"),
            ("KGV (P/E)", info.get('trailingPE', 'N/A'), ""),
            ("KUV (P/S)", info.get('priceToSalesTrailing12Months', 'N/A'), ""),
            ("Marktkapitalisierung", f"{info.get('marketCap', 0) / 1e9:.2f}" if info.get('marketCap') else 'N/A',
             "Mrd. $"),
            ("52W Hoch", info.get('fiftyTwoWeekHigh', 'N/A'), "$"),
            ("52W Tief", info.get('fiftyTwoWeekLow', 'N/A'), "$"),
            ("Beta (5J)", info.get('beta', 'N/A'), ""),
            ("Dividendenrendite", f"{info.get('dividendYield', 0) * 100:.2f}" if info.get('dividendYield') else 'N/A',
             "%"),
            ("Free Cash Flow", f"{info.get('freeCashflow', 0) / 1e9:.2f}" if info.get('freeCashflow') else 'N/A',
             "Mrd. $"),
            ("Debt/Equity", f"{info.get('debtToEquity', 0):.2f}" if info.get('debtToEquity') else 'N/A', ""),
            ("Short % Float",
             f"{info.get('shortPercentOfFloat', 0) * 100:.2f}%" if info.get('shortPercentOfFloat') else 'N/A', ""),
            ("Inst. Besitz",
             f"{info.get('heldPercentInstitutions', 0) * 100:.1f}%" if info.get('heldPercentInstitutions') else 'N/A',
             "")
        ]

        for i, (name, value, unit) in enumerate(metrics):
            cols[i % 4].metric(name, f"{value} {unit}" if value != 'N/A' else value)

        # Dividendenanalyse
        if 'dividendRate' in info and info['dividendRate'] > 0:
            st.subheader("💵 Dividendenanalyse")
            cols = st.columns(4)
            cols[0].metric("Aktuelle Dividende", f"${info['dividendRate']:.2f}")
            if 'dividendYield' in info:
                cols[1].metric("Dividendenrendite", f"{info['dividendYield'] * 100:.2f}%")
            if 'payoutRatio' in info:
                cols[2].metric("Ausschüttungsquote", f"{info['payoutRatio'] * 100:.1f}%")
            if 'dividendDate' in info:
                dividend_date = datetime.fromtimestamp(info['dividendDate']).strftime('%d.%m.%Y')
                cols[3].metric("Letzter Dividendentermin", dividend_date)

        # Handelsvolumen & Liquidität
        st.subheader("💎 Handelsdaten")
        cols = st.columns(3)
        avg_volume = hist_data['Volume'].mean() / 1e6
        latest_volume = hist_data['Volume'].iloc[-1] / 1e6
        cols[0].metric("Durchschn. Volumen (30T)",
                       f"{avg_volume:.1f} Mio.",
                       f"Letzter Tag: {latest_volume:.1f} Mio.")

        if 'averageVolume' in info:
            cols[1].metric("Durchschn. Volumen (10T)",
                           f"{info['averageVolume'] / 1e6:.1f} Mio.")

        if 'volume' in info:
            cols[2].metric("Aktuelles Volumen",
                           f"{info['volume'] / 1e6:.1f} Mio.")

        # Kursverlauf Chart
        st.subheader(f"📈 Kursverlauf: {ticker}")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(hist_data.index, hist_data['Close'], label='Schlusskurs', color='blue')
        ax.set_title(f'Kursverlauf der letzten 5 Jahre', fontsize=16)
        ax.set_xlabel('Datum', fontsize=12)
        ax.set_ylabel('Preis ($)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        st.pyplot(fig)

        # Monte Carlo Simulation
        st.subheader("🔮 Monte Carlo Prognose (nächste 30 Tage)")
        mc_results = monte_carlo_simulation(hist_data)

        avg_forecast = mc_results.mean(axis=1)
        high_forecast = mc_results.max(axis=1)
        low_forecast = mc_results.min(axis=1)
        mid_high = (avg_forecast + high_forecast) / 2
        mid_low = (avg_forecast + low_forecast) / 2

        forecast_dates = [hist_data.index[-1] + timedelta(days=i) for i in range(1, 31)]

        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(forecast_dates, avg_forecast, label='Durchschnitt', color='blue')
        ax2.plot(forecast_dates, high_forecast, label='Maximum', color='green', linestyle=':')
        ax2.plot(forecast_dates, low_forecast, label='Minimum', color='red', linestyle=':')
        ax2.plot(forecast_dates, mid_high, label='Mittleres Hoch', color='lightgreen', linestyle='--')
        ax2.plot(forecast_dates, mid_low, label='Mittleres Tief', color='salmon', linestyle='--')

        ax2.fill_between(forecast_dates, low_forecast, high_forecast, color='gray', alpha=0.1)
        ax2.set_title('30-Tage-Preisprognose (Monte Carlo Simulation)', fontsize=16)
        ax2.set_xlabel('Datum', fontsize=12)
        ax2.set_ylabel('Prognostizierter Preis ($)', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend()
        st.pyplot(fig2)

        # Prognose Zusammenfassung
        st.subheader("📌 Prognose Zusammenfassung")
        summary_cols = st.columns(3)

        summary_data = [
            ("Durchschnittlicher Endpreis", avg_forecast[-1]),
            ("Mittleres Hoch", mid_high[-1]),
            ("Maximaler Preis", high_forecast[-1]),
            ("Aktueller Preis", hist_data['Close'].iloc[-1]),
            ("Minimaler Preis", low_forecast[-1]),
            ("Mittleres Tief", mid_low[-1])
        ]

        for i, (label, value) in enumerate(summary_data):
            summary_cols[i % 3].metric(label, f"${value:.2f}")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("ℹ️ Daten von Yahoo Finance")