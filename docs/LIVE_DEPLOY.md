# Bybit linear cross-section: live handoff

Cloud does not place orders, change leverage, transfer funds, or start the service. Host start is `crypto-xs-live --supervise` after the gates below. This is not an exclusive-account claim.

## Owner acceptance, 2026-11-07 22:48

The owner accepted a software 100 USDT budget on UID `211344229` and does not require physical segregation. XAUT PA and mm-shadow@bybit were disabled and masked at 22:49:47 MYT (inactive, PID 0). A 22:50:24 MYT read showed that UID flat, pagination complete, free USDT 121.50290183, borrow 0. Those facts do not reserve 100 USDT and do not make the account exclusive.

`accept_shared_master_uid: true` records that acceptance. It is not a certificate that no other key can trade this UID. Foreign positions and orders still block new entries and are not cancelled. Copy `config/crypto_xs_live.accepted-budget.json` to `/etc/super-trader/crypto_xs_live.json`. Put `SUPER_TRADER_LIVE=1` and the credential file paths, not the secrets, in `/etc/super-trader/crypto-xs.env`. Do not set `SUPER_TRADER_OFFLINE_FIXTURE` there; that variable is only for the process test and never reaches Bybit.

## Supported process

The pt host has `python3` and does not have `/usr/bin/python`. The unit therefore starts `/opt/super-trader/.venv/bin/python`, created from `python3`:

```sh
sudo mkdir -p /opt/super-trader
sudo rsync -a ./ /opt/super-trader/
sudo sh deploy/install-supervisor-venv.sh /opt/super-trader
test -x /opt/super-trader/.venv/bin/python
/opt/super-trader/.venv/bin/python -m super_trader crypto-xs-live --config /etc/super-trader/crypto_xs_live.json --preflight
```

`deploy/crypto-xs-supervisor.service` is the start and stop path. Its `ExecStart` is that venv interpreter. Do not point it at `/usr/bin/python`. Do not enable the old refused one-shot unit; that unit cannot supervise.

```sh
sudo systemctl enable --now crypto-xs-supervisor.service
sudo systemctl stop crypto-xs-supervisor.service
```

Exit 2 means the three opt-in switches are absent or the ledger lock is held. systemd does not restart exit 2. Exit 1 means three account reads or cycle errors in a row; the process has stopped opening. Exit 0 is a clean SIGTERM.

The loop waits 60 seconds between passes. Timestamps are UTC. A Bybit linear daily bar opens at 00:00 UTC and is usable only after the next 00:00 UTC; the forming bar is dropped. `--supervise` submits a basket only for a new closed bar. Between bars it only checks this strategy's own PnL. A loss of 20 USDT cancels this strategy's orders, reduce-only closes only its quantity, and the halt stays in the ledger across restarts. `--monitor` remains a single stale pass and does not force a daily order.

## Remaining order gate

LINKUSDT, AVAXUSDT, and SUIUSDT were last read at 10x. This program still refuses new entries until a fresh read shows leverage 1 and `positionIdx` 0. It will not send the leverage change. The symbol-only calls are below. Until that read succeeds, `--supervise` stays up, writes `entries_blocked: true`, and sends no opening order.

A later full pagination of LINKUSDT contained two rows for position index 0: one `Normal` row with leverage 1 and timestamps, and one later page with leverage 10 and empty status, times, and mark. Bybit documents one slot per `positionIdx` and does not define which duplicate wins. The adapter does not drop either row and does not let the last page replace the first. Unequal leverage on that slot leaves the symbol unknown, so the 1x gate stays closed. AVAX and SUI were not changed and remain 10x.

The 23:13:17 MYT capture shows why those two rows are not yet a settled conflict. Page 1 returned `nextPageCursor=LINKUSDT%2C1794063867198%2C0`. Page 2 was sent as `cursor=LINKUSDT%252C1794063867198%252C0` because `urllib.parse.urlencode` encoded the `%` again. Official pybit copies that cursor into the query string without a second encoding, and the signature is over that same string. This client now does that. The `settleCoin=USDT&limit=200` call returned an empty list, which matches the documented non-zero-size filter and does not decide LINK's leverage.

Repeat this read-only pair. The second URL must contain `%2C`, not `%252C`:

```text
GET /v5/position/list?category=linear&symbol=LINKUSDT
GET /v5/position/list?category=linear&cursor=LINKUSDT%2C1794063867198%2C0&symbol=LINKUSDT
```

If that full result still has two different leverages for position index 0, the 1x gate stays closed. This package does not drop either of the already captured rows to force a pass.

## Unresolved wallet risk

UID `211344229` is the same master account whose key was in the XAUT PA process. That process is stopped and its unit is masked; this file does not say the UID is exclusive. The 22:50:24 MYT book was empty, with free USDT 121.50290183 and borrow 0. A later order on this UID can still use that free balance. Super_Trader will then block new entries and will not cancel the other order.

Isolated margin does not reserve 100 USDT from other strategies. The key cannot transfer or withdraw, so this program cannot move 100 USDT into a subaccount. The 100 USDT figure is only a virtual budget: orders are sized from `min(100, strategy equity, free USDT)`, never from the 121.50 wallet and never multiplied by the exchange leverage.

Every cycle reads positions and orders across linear USDT, linear USDC, inverse, option, and spot, including conditional filters where the API allows them. Any foreign position or order blocks new risk. This program does not cancel or adopt it. That still loses the race if another bot uses the free balance after the check.

LINKUSDT, AVAXUSDT, and SUIUSDT were at 10x leverage and one-way mode. Nothing was changed. New entries stay off until each of those symbols shows leverage 1. Setting that leverage is a separate, explicitly authorized operator step. It is not run by this repository.

## Frozen universe

Trade only these linear USDT perpetuals:

- LINKUSDT, minimum quantity 0.1, step 0.1, minimum notional 5
- AVAXUSDT, minimum quantity 0.1, step 0.1, minimum notional 5
- SUIUSDT, minimum quantity 10, step 10, minimum notional 5

The read-only check saw 365 daily rows and a possibly incomplete current candle. The runner drops a daily bar until its UTC close time has passed.

Forbidden underlyings, taken from the existing bot configs and kept even if a config file tries to delete them: ADA, BTC, DOGE, ETH, SOL, XAUT, XOM, XRP. The block covers the USDT, USDC, USD, and PERP symbols for each of them. XAUT and SOL stay excluded. No performance is claimed for the three names that remain.

The basket is one long and one short. Gross notional cannot exceed the virtual budget. If quantity steps cannot form that pair without crossing the cap or leaving more than 10 USDT of imbalance, the cycle sends nothing.

## Symbol-only 1x leverage, not a global change

Do not run this from the cloud agent. Do not call account-mode, margin-mode, or leverage endpoints from `crypto-xs-live`. After a separate authorization, an operator can set only these three symbols, one call each:

```json
POST /v5/position/set-leverage
{"category":"linear","symbol":"LINKUSDT","buyLeverage":"1","sellLeverage":"1"}
```

Repeat with `AVAXUSDT` and `SUIUSDT`. There is no account-wide leverage switch here. Re-read the positions afterward and confirm `leverage` is `1` and `positionIdx` is `0` before enabling entries. This code rejects `/v5/position/set-leverage`, `/v5/order/cancel-all`, and transfer paths if something tries to call them.

## Loss, ownership, and collateral

Strategy PnL is realized fills, fees, funding attributed to symbols this strategy has traded, and unrealized marks on its own quantity. A wallet deposit or the extra 21.50 USDT is not PnL. At or below -20 USDT the ledger halts, cancels this strategy's own `orderLinkId`s, and sends reduce-only orders only for its own quantity on symbols that are not mixed with a foreign position. A later restart stays halted. The close can fill worse than -20 USDT.

`orderLinkId` starts with `cxls`. The intent is written to `data/crypto_xs_live/ledger.json` before the request. A timeout keeps that id and does not invent another one.

Free collateral uses `walletBalance - totalPositionIM - totalOrderIM - locked - bonus`. A blank field is unknown, not zero. Required margin is treated as 1x notional even if the exchange is set to 10x. Borrowed USDT blocks new entries.

## Commands

Credentials stay in `BYBIT_API_KEY` and `BYBIT_API_SECRET`, or in `BYBIT_API_KEY_FILE` and `BYBIT_API_SECRET_FILE`. Do not put them in the config, the ledger, or logs.

```powershell
python -m super_trader crypto-xs-live --config config/crypto_xs_live.example.json --preflight
```

That example has `live_enabled` and `accept_shared_master_uid` set false, so it exits 2 and does not open a private connection. Copy it to `config/crypto_xs_live.json` only on the deployment host. Turn the two flags on only after leverage is 1 and the shared-UID limit is accepted. Then:

```powershell
set SUPER_TRADER_LIVE=1
python -m super_trader crypto-xs-live --config config/crypto_xs_live.json --monitor
python -m super_trader crypto-xs-live --config config/crypto_xs_live.json --once
```

`--monitor` checks the account and can reduce or halt. `--once` also ranks the latest closed UTC day. Repeat a read-only preflight immediately before any real start. This cloud build has not done that start.

Paper research remains `python -m super_trader crypto-xs`. Its `--live` flag still refuses.
