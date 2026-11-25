from __future__ import annotations

import numpy as np
import pandas as pd


def monthly_discount_rate(config: dict) -> float:
    return (1 + config["discount_rate_annual"]) ** (1 / config["months_per_year"]) - 1

def calculate_scenarios(config: dict, cpm: float | None = None, churn_multiplier: float = 1.0) -> pd.DataFrame:
    users = config["active_users"]
    arpu = config["subscription_arpu_rub_month"]
    cpm_value = config["cpm_rub"] if cpm is None else cpm
    cac = config["cac_rub"]
    lifetime = config["lifetime_months"]
    days = config["days_per_month"]
    per_thousand = config["cpm_impressions"]
    loads = np.asarray(config["load_levels_impressions_day"], dtype=float)
    churn = np.clip(np.asarray(config["churn_rates"], dtype=float) * churn_multiplier, 0, 1)
    watch = np.asarray(config["watch_time_minutes"], dtype=float)
    advertising = users * loads * days * cpm_value / per_thousand
    churned = users * churn
    retained = users - churned
    subscription = retained * arpu
    replacement_cac = churned * cac
    ltv_opportunity_loss = churned * arpu * lifetime
    total_ltv_loss = replacement_cac + ltv_opportunity_loss
    cash_profit = advertising + subscription - replacement_cac
    economic_profit_after_ltv = advertising + subscription - total_ltv_loss
    return pd.DataFrame({
        "Нагрузка, показов/день": loads.astype(int),
        "Отток, %": churn * 100,
        "Время просмотра, мин": watch,
        "Сохранившиеся пользователи": retained,
        "Ушедшие пользователи": churned,
        "Доход рекламы, руб.": advertising,
        "Доход подписки, руб.": subscription,
        "Потери CAC, руб.": replacement_cac,
        "Упущенный LTV, руб.": ltv_opportunity_loss,
        "Потери оттока с LTV, руб.": total_ltv_loss,
        "Денежный поток, руб.": cash_profit,
        "Экономический результат с LTV, руб.": economic_profit_after_ltv,
    })

def npv_for_scenario(config: dict, monthly_cash_flow: float) -> float:
    r = monthly_discount_rate(config)
    horizon = config["analysis_horizon_months"]
    initial = config["initial_investment_rub"]
    return -initial + sum(monthly_cash_flow / ((1 + r) ** month) for month in range(1, horizon + 1))

def add_npv_columns(config: dict, df: pd.DataFrame) -> pd.DataFrame:
    out=df.copy()
    out["NPV денежного потока, руб."]=out["Денежный поток, руб."].map(lambda x: npv_for_scenario(config,float(x)))
    out["NPV с учётом LTV, руб."]=out["Экономический результат с LTV, руб."].map(lambda x: npv_for_scenario(config,float(x)))
    return out

def best_scenario(df: pd.DataFrame, metric: str = "NPV денежного потока, руб.") -> pd.Series:
    return df.loc[df[metric].idxmax()]

def format_rub(value: float, decimals: int = 0) -> str:
    return f"{value:,.{decimals}f}".replace(",", " ").replace(".", ",")

def format_mln(value: float, decimals: int = 1) -> str:
    return format_rub(value / 1_000_000, decimals)


def calculate_cohort_scenarios(config: dict) -> pd.DataFrame:
    cohort = config["cohort_model"]
    users = config["active_users"]
    days = config["days_per_month"]
    per_thousand = config["cpm_impressions"]
    cpm = config["cpm_rub"]
    cac = config["cac_rub"]
    loads = np.asarray(cohort["loads_impressions_day"], dtype=float)
    base = calculate_scenarios(config)
    churn = np.asarray(config["churn_rates"], dtype=float)
    watch = np.asarray(cohort["watch_time_minutes"], dtype=float)
    ctr = np.asarray(cohort["ctr_rates"], dtype=float)
    conversion = np.asarray(cohort["paid_conversion_rates"], dtype=float)
    paid_arpu = np.asarray(cohort["paid_arpu_rub_month"], dtype=float)
    retained = users * (1 - churn)
    churned = users * churn
    impressions = users * loads * days
    clicks = impressions * ctr
    paid_users = retained * conversion
    ads = impressions * cpm / per_thousand
    subscription = paid_users * paid_arpu
    cash_flow = ads + subscription - churned * cac
    return pd.DataFrame({
        "Нагрузка, показов/день": loads.astype(int),
        "Время просмотра, мин": watch,
        "CTR, %": ctr * 100,
        "Конверсия в подписку, %": conversion * 100,
        "Платные пользователи": paid_users,
        "Клики": clicks,
        "Доход рекламы, руб.": ads,
        "Когортная подписная выручка, руб.": subscription,
        "Денежный поток, руб.": cash_flow,
        "NPV денежного потока, руб.": [npv_for_scenario(config, x) for x in cash_flow],
        "Отток, %": churn * 100,
    })
