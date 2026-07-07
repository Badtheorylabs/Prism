"""Benchmark task fixtures (shared).

Each Task is a tiny repo whose tests define success. Used by layer_bench.py
(the layer-delta benchmark that isolates Prism's contribution). The old
final-code-resolution benchmark that used these has been removed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Task:
    name: str
    prompt: str
    files: dict[str, str]  # path -> content (includes the grading tests/)
    band: str = "basic"    # basic | recoverable | context


TASKS: list[Task] = [
    Task(
        "impl_add",
        "Implement calc.add(a, b) to return the sum of a and b.",
        {
            "calc.py": "def add(a, b):\n    raise NotImplementedError\n",
            "tests/test_calc.py": (
                "from calc import add\n\n"
                "def test_add():\n    assert add(2, 3) == 5\n    assert add(-1, 1) == 0\n"
            ),
        },
    ),
    Task(
        "fix_reverse",
        "Fix reverse_words so it reverses the ORDER of the words in the string.",
        {
            "strutil.py": "def reverse_words(s):\n    return ' '.join(s.split())  # bug: no reversal\n",
            "tests/test_strutil.py": (
                "from strutil import reverse_words\n\n"
                "def test_reverse():\n    assert reverse_words('a b c') == 'c b a'\n"
            ),
        },
        band="recoverable",
    ),
    Task(
        "empty_mean",
        "Make mean([]) return 0 instead of raising, keeping normal means correct.",
        {
            "mathx.py": "def mean(xs):\n    return sum(xs) / len(xs)\n",
            "tests/test_mathx.py": (
                "from mathx import mean\n\n"
                "def test_mean():\n    assert mean([2, 4]) == 3\n    assert mean([]) == 0\n"
            ),
        },
        band="recoverable",
    ),
    Task(
        "cross_file_hash",
        "Make auth.hash_password NOT store the plaintext (transform it), while keeping api.verify working.",
        {
            "auth.py": "def hash_password(p):\n    return p  # stores plaintext\n",
            "api.py": (
                "from auth import hash_password\n\n"
                "def verify(p, stored):\n    return hash_password(p) == stored\n"
            ),
            "tests/test_api.py": (
                "from api import verify\nfrom auth import hash_password\n\n"
                "def test_hash():\n"
                "    h = hash_password('secret')\n"
                "    assert h != 'secret'\n"
                "    assert verify('secret', h)\n"
                "    assert not verify('nope', h)\n"
            ),
        },
        band="recoverable",
    ),
    Task(
        "rate_limit",
        "Implement Limiter.allow so it allows at most n calls per key, then blocks further calls.",
        {
            "limits.py": (
                "class Limiter:\n"
                "    def __init__(self, n):\n        self.n = n\n        self.hits = {}\n\n"
                "    def allow(self, key):\n        return True  # TODO: enforce the limit\n"
            ),
            "tests/test_limits.py": (
                "from limits import Limiter\n\n"
                "def test_limit():\n"
                "    lim = Limiter(2)\n"
                "    assert lim.allow('u')\n    assert lim.allow('u')\n"
                "    assert not lim.allow('u')\n"
                "    assert lim.allow('other')\n"
            ),
        },
    ),
]


RECOVERABLE_TASKS: list[Task] = [
    Task(
        "clamp_edges",
        "Fix clamp(x, lo, hi) so it respects the lower and upper bounds.",
        {
            "bounds.py": (
                "def clamp(x, lo, hi):\n"
                "    if x > hi:\n"
                "        return hi\n"
                "    return x  # bug: lower bound ignored\n"
            ),
            "tests/test_bounds.py": (
                "from bounds import clamp\n\n"
                "def test_clamp_edges():\n"
                "    assert clamp(5, 0, 10) == 5\n"
                "    assert clamp(99, 0, 10) == 10\n"
                "    assert clamp(-4, 0, 10) == 0\n"
            ),
        },
        band="recoverable",
    ),
    Task(
        "parse_int_default",
        "Make parse_count return an integer count, falling back to the default for invalid input.",
        {
            "parser.py": "def parse_count(raw, default=0):\n    return int(raw)\n",
            "tests/test_parser.py": (
                "from parser import parse_count\n\n"
                "def test_parse_count():\n"
                "    assert parse_count('7') == 7\n"
                "    assert parse_count(' 8 ') == 8\n"
                "    assert parse_count('', default=3) == 3\n"
                "    assert parse_count('many', default=2) == 2\n"
            ),
        },
        band="recoverable",
    ),
    Task(
        "unique_preserve_order",
        "Fix unique so it removes duplicates but preserves first-seen order.",
        {
            "dedupe.py": "def unique(items):\n    return list(set(items))\n",
            "tests/test_dedupe.py": (
                "from dedupe import unique\n\n"
                "def test_unique_preserves_order():\n"
                "    assert unique(['b', 'a', 'b', 'c', 'a']) == ['b', 'a', 'c']\n"
                "    assert unique([]) == []\n"
            ),
        },
        band="recoverable",
    ),
    Task(
        "paginate_one_based",
        "Fix paginate so page numbers are one-based and per_page controls the window size.",
        {
            "pages.py": (
                "def paginate(items, page, per_page):\n"
                "    start = page * per_page\n"
                "    return items[start:start + per_page]\n"
            ),
            "tests/test_pages.py": (
                "from pages import paginate\n\n"
                "def test_paginate_one_based():\n"
                "    data = list(range(10))\n"
                "    assert paginate(data, 1, 3) == [0, 1, 2]\n"
                "    assert paginate(data, 2, 3) == [3, 4, 5]\n"
                "    assert paginate(data, 4, 3) == [9]\n"
            ),
        },
        band="recoverable",
    ),
    Task(
        "expiry_boundary",
        "Fix is_expired so a token expiring exactly now is considered expired.",
        {
            "tokens.py": "def is_expired(now, expires_at):\n    return now > expires_at\n",
            "tests/test_tokens.py": (
                "from tokens import is_expired\n\n"
                "def test_expiry_boundary():\n"
                "    assert not is_expired(9, 10)\n"
                "    assert is_expired(10, 10)\n"
                "    assert is_expired(11, 10)\n"
            ),
        },
        band="recoverable",
    ),
    Task(
        "moving_average_window",
        "Fix moving_average so it returns each full sliding-window average.",
        {
            "series.py": (
                "def moving_average(xs, window):\n"
                "    out = []\n"
                "    for i in range(len(xs) - window):\n"
                "        out.append(sum(xs[i:i + window]) / window)\n"
                "    return out\n"
            ),
            "tests/test_series.py": (
                "from series import moving_average\n\n"
                "def test_moving_average_full_windows():\n"
                "    assert moving_average([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]\n"
                "    assert moving_average([10, 20, 30], 3) == [20]\n"
            ),
        },
        band="recoverable",
    ),
    Task(
        "retry_backoff_cap",
        "Fix retry_delay so retries use exponential backoff with a maximum cap.",
        {
            "retry.py": (
                "def retry_delay(attempt, base=1, cap=30):\n"
                "    return min(cap, base * attempt)\n"
            ),
            "tests/test_retry.py": (
                "from retry import retry_delay\n\n"
                "def test_retry_delay_exponential_with_cap():\n"
                "    assert retry_delay(0) == 1\n"
                "    assert retry_delay(1) == 2\n"
                "    assert retry_delay(3) == 8\n"
                "    assert retry_delay(10, base=2, cap=20) == 20\n"
            ),
        },
        band="recoverable",
    ),
    Task(
        "inventory_transfer",
        "Fix transfer so it moves stock between warehouses and rejects insufficient stock.",
        {
            "inventory.py": (
                "class Warehouse:\n"
                "    def __init__(self, stock):\n"
                "        self.stock = dict(stock)\n\n"
                "    def count(self, sku):\n"
                "        return self.stock.get(sku, 0)\n\n"
                "    def remove(self, sku, qty):\n"
                "        self.stock[sku] = self.count(sku) - qty\n\n"
                "    def add(self, sku, qty):\n"
                "        self.stock[sku] = self.count(sku) + qty\n"
            ),
            "ops.py": (
                "def transfer(src, dst, sku, qty):\n"
                "    src.remove(sku, qty)\n"
                "    return True\n"
            ),
            "tests/test_ops.py": (
                "from inventory import Warehouse\n"
                "from ops import transfer\n\n"
                "def test_transfer_moves_stock():\n"
                "    a = Warehouse({'book': 3})\n"
                "    b = Warehouse({})\n"
                "    assert transfer(a, b, 'book', 2)\n"
                "    assert a.count('book') == 1\n"
                "    assert b.count('book') == 2\n\n"
                "def test_transfer_rejects_insufficient_stock():\n"
                "    a = Warehouse({'book': 1})\n"
                "    b = Warehouse({})\n"
                "    assert not transfer(a, b, 'book', 2)\n"
                "    assert a.count('book') == 1\n"
                "    assert b.count('book') == 0\n"
            ),
        },
        band="recoverable",
    ),
    Task(
        "slugify_punctuation",
        "Fix slugify so it creates clean lowercase URL slugs.",
        {
            "slug.py": "def slugify(title):\n    return title.lower().replace(' ', '-')\n",
            "tests/test_slug.py": (
                "from slug import slugify\n\n"
                "def test_slugify_clean_url_slug():\n"
                "    assert slugify('Hello, World!') == 'hello-world'\n"
                "    assert slugify('  A  B  ') == 'a-b'\n"
                "    assert slugify('API_v2 ready') == 'api-v2-ready'\n"
            ),
        },
        band="recoverable",
    ),
    Task(
        "cart_totals",
        "Fix cart_total so discounts apply before tax and shipping is added once.",
        {
            "cart.py": (
                "def cart_total(items, discount=0, tax=0, shipping=0):\n"
                "    subtotal = sum(item['price'] * item.get('qty', 1) for item in items)\n"
                "    return subtotal + subtotal * tax - discount + shipping * len(items)\n"
            ),
            "tests/test_cart.py": (
                "from cart import cart_total\n\n"
                "def test_cart_total_order_of_operations():\n"
                "    items = [{'price': 10, 'qty': 2}, {'price': 5, 'qty': 1}]\n"
                "    assert cart_total(items, discount=5, tax=0.1, shipping=3) == 25\n"
                "    assert cart_total([], discount=5, tax=0.1, shipping=3) == 3\n"
            ),
        },
        band="recoverable",
    ),
]


def _noise_files(n: int = 48) -> dict[str, str]:
    files: dict[str, str] = {}
    for i in range(n):
        files[f"noise/module_{i:02d}.py"] = (
            f"def helper_{i}(value):\n"
            f"    return value + {i}\n\n"
            f"class Noise{i}:\n"
            f"    def compute(self, x):\n"
            f"        return helper_{i}(x)\n"
        )
    return files


CONTEXT_TASKS: list[Task] = [
    Task(
        "noisy_discount_context",
        "Fix final_price so VIP users get the configured discount before tax.",
        {
            **_noise_files(),
            "billing/pricing.py": (
                "from billing.rules import discount_for\n\n"
                "def final_price(user, subtotal, tax_rate):\n"
                "    discount = 0\n"
                "    taxable = subtotal - discount\n"
                "    return round(taxable + taxable * tax_rate, 2)\n"
            ),
            "billing/rules.py": (
                "def discount_for(user):\n"
                "    if user.get('tier') == 'vip':\n"
                "        return 0.20\n"
                "    return 0.0\n"
            ),
            "tests/test_pricing.py": (
                "from billing.pricing import final_price\n\n"
                "def test_vip_discount_before_tax():\n"
                "    assert final_price({'tier': 'vip'}, 100, 0.10) == 88.0\n"
                "    assert final_price({'tier': 'basic'}, 100, 0.10) == 110.0\n"
            ),
        },
        band="context",
    ),
]

TASKS.extend(RECOVERABLE_TASKS)
TASKS.extend(CONTEXT_TASKS)

TASK_BY_NAME = {t.name: t for t in TASKS}


def materialize(task: Task, root: str) -> None:
    for path, content in task.files.items():
        dst = os.path.join(root, path)
        os.makedirs(os.path.dirname(dst) or root, exist_ok=True)
        with open(dst, "w", encoding="utf-8") as fh:
            fh.write(content)


def _visible_source(root: str) -> str:
    """Concatenate non-test source for the naive baseline prompt."""
    chunks = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (".prism", "__pycache__", "tests")]
        for fn in sorted(filenames):
            if fn.endswith(".py"):
                p = os.path.join(dirpath, fn)
                rel = os.path.relpath(p, root)
                with open(p, encoding="utf-8") as fh:
                    chunks.append(f"FILE: {rel}\n{fh.read()}")
    return "\n\n".join(chunks)
