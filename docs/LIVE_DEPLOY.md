# Bybit linear cross-section: live handoff

This build does not place orders, change leverage, transfer funds, or deploy a service. The cloud environment did not call the account. Facts below are the sanitized read-only preflight from the approved host at 2026-11-07 22:10 MYT.

## Unresolved isolation blocker

UID `211344229` is a mainnet master UTA2 account in `ISOLATED_MARGIN`. At the snapshot it had no positions and no ordinary or conditional orders, with 121.50290183 USDT free. That does not make the account dedicated. Other bots' keys were not compared with this UID, and an empty book can change later.

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
