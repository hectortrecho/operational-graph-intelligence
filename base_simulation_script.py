#!/bin/env python3 
import pandas as pd
import numpy as np

np.random.seed(42) # Verdad absoluta del universo.

# Parametros
n = 100000

regions = ['México', 'Chile', 'Brazil', 'Argentina', 'Colombia','Perú'] # principales mercados de latam.
case_types = ['Order Delay', 'Pricing Issue', 'Documentation', 'Technical Support', 'Order modifications','Order entry']
customer_types = ['Distributor', 'End Customer']

# Genera información base.
data = pd.DataFrame({
    'case_id': range(1000, 1000+n),
    'region': np.random.choice(regions, n),
    'case_type': np.random.choice(case_types, n),
    'customer_type': np.random.choice(customer_types, n),
    'created_date': pd.to_datetime('2026-01-01') + pd.to_timedelta(np.random.randint(0, 90, n), unit='D')
})

# Backlog simulado en dias (Influenciado por region)
region_delay_factor = {
    'Mexico': 2,
    'Chile': 3,
    'Brazil': 5,
    'Argentina': 4
}

data['backlog_days'] = data['region'].map(region_delay_factor) + np.random.randint(0, 5, n)

# Tiempo de resolution depende del backlog + aleatoriedad.
data['resolution_days'] = data['backlog_days'] + np.random.randint(1, 4, n)

# Regla de acuerdo de nivel de servicio (SLA)
SLA_LIMIT = 7
data['sla_breach'] = data['resolution_days'] > SLA_LIMIT

# Genera fechas de resolución.
data['resolved_date'] = data['created_date'] + pd.to_timedelta(data['resolution_days'], unit='D')

# Probabilidad de escalar un incidente depende del backlog.
data['escalated'] = np.where(data['backlog_days'] > 5,
                             np.random.choice([0,1], n, p=[0.6,0.4]),
                             np.random.choice([0,1], n, p=[0.85,0.15]))

# Temporalidad y comportamiento de cliente:
data['month'] = data['created_date'].dt.month

# Simulando backlog de respuesta en ciertos meses.
data['seasonal_factor'] = data['month'].apply(lambda x: 3 if x in [3,4] else 1)

data['backlog_days'] = data['backlog_days'] + data['seasonal_factor']
customer_behavior = {
    'Distributor': 2,
    'End Customer': 1
}

data['complexity_score'] = data['customer_type'].map(customer_behavior) + np.random.randint(0,3,n)

# Simulando  comentarios de clientes:
comments = [
    "Customer is requesting urgent update",
    "Delay due to logistics issue",
    "Pricing discrepancy identified",
    "Awaiting internal approval",
    "Customer escalated the issue"
]

data['case_notes'] = np.random.choice(comments, n)

data.to_csv("customer_support_simulated.csv", index=False) # exportando datos.

print(data.head())
