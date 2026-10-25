"""One schema for the forecast ledger, Markdown table and plotted month windows."""
from datetime import date
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'docs/forecasts.json'
TABLE_START = '<!-- forecast-table:start -->'
TABLE_END = '<!-- forecast-table:end -->'
CONFIDENCE = {'高', '中高', '中'}


def month_label(as_of, offset):
    anchor = date.fromisoformat(as_of)
    month = anchor.year * 12 + anchor.month - 1 + offset
    year, month = divmod(month, 12)
    return f'{year:04d}-{month+1:02d}'


def validate_forecasts(data):
    date.fromisoformat(data['as_of'])
    reviewed = date.fromisoformat(data['reviewed_on'])
    if reviewed < date.fromisoformat(data['as_of']):
        raise ValueError('Review date predates forecast origin')
    rows = data['forecasts']
    if not rows or len({f['id'] for f in rows}) != len(rows):
        raise ValueError('Forecast IDs must be nonempty and unique')
    for row in rows:
        for key in ['id', 'topic', 'plot_label', 'prediction', 'baseline', 'scope',
                    'confirmation', 'falsifier', 'confidence_reason']:
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"{row.get('id')}: missing {key}")
        if row['confidence'] not in CONFIDENCE:
            raise ValueError('Unknown qualitative confidence')
        if not row.get('basis') or not row.get('success_criteria'):
            raise ValueError('Evidence and measurable criteria required')
        window = row['window_months_after_as_of']
        if len(window) != 2 or any(type(x) is not int or x < 0 for x in window) or window[1] <= window[0]:
            raise ValueError('Need increasing nonnegative integer month window')
        horizon = ' 至 '.join(month_label(data['as_of'], x) for x in window)
        if row['horizon'] != horizon:
            raise ValueError(f"{row['id']}: date labels do not match month window")
        if row['status'] != 'pending':
            raise ValueError('This ledger has not yet been adjudicated')
    return data


def load_forecasts(path=LEDGER):
    return validate_forecasts(json.loads(path.read_text(encoding='utf-8')))


def render_table(data):
    validate_forecasts(data)
    def cell(value):
        return value.replace('|', '\\|').replace('\n', ' ')
    lines = [TABLE_START,
             '| 编号／方向 | 评估窗口 | 信心 | 应发生的新增事件 | 判定要点 |',
             '|---|---|---|---|---|']
    for row in data['forecasts']:
        values = [row['id']+' '+row['topic'], row['horizon'], row['confidence'],
                  row['prediction'], row['confirmation']]
        lines.append('| '+' | '.join(cell(value) for value in values)+' |')
    return '\n'.join([*lines, TABLE_END])
