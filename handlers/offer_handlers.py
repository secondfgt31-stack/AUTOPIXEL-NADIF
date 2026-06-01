"""Offer detection and 2FA-related handlers."""

import asyncio
import logging
import random
import time

from telegram import Update
from telegram.ext import ContextTypes, ConversationHandler

import config
from core.proxy_manager import PROXY_MANAGER, mask_proxy_url, parse_proxy_parts
from core.session_manager import (
    SESSION_STORE,
    ensure_proxy_session_token,
    get_session,
    secure_wipe,
)
from services.device_simulator import create_device_profile
from services.google_automation import (
    GoogleAutomationError,
    check_offer_with_driver,
    close_driver,
    diagnose_offer_page,
    dump_offer_debug_artifacts,
    resolve_manual_login,
    start_login,
    submit_2fa_code,
)
from services.network_diagnostics import (
    format_connection_identity,
    inspect_connection,
    probe_google_signin,
)
from handlers.ui import (
    main_menu_keyboard,
    prepare_action_message,
    quick_actions_inline_keyboard,
    tr,
)

logger = logging.getLogger(__name__)

AWAIT_2FA_CODE = 10
AWAIT_MANUAL_VERIFICATION = 11
LAST_CHECK_TIME: dict[int, float] = {}
CHECK_OFFER_COOLDOWN = 10  # 5 minutes between checks per user
CHROME_SEMAPHORE = asyncio.Semaphore(1)
DIAGNOSTIC_I18N_KEYS = {
    (
        "Google One shows AI-related products, but the promo state is mixed "
        "and needs manual review."
    ): "offer_diag_ai_mixed",
    (
        "Google One shows regular paid Google AI Pro plans for this account, "
        "but no free promo claim link was present."
    ): "offer_diag_paid_no_free",
    "Google One loaded your normal account plan page, but no promo card was present.": (
        "offer_diag_normal_plan_no_promo"
    ),
    (
        "Google One shows an embedded Google AI trial offer on the plans page, "
        "but AutoPixel did not capture the checkout link automatically."
    ): "offer_diag_embedded_ai_trial",
    (
        "Google One shows an embedded free-trial offer on the plans page, "
        "but AutoPixel did not capture the checkout link automatically."
    ): "offer_diag_embedded_trial",
}


def _clear_pending_verification(session: dict) -> None:
    """Drop temporary verification metadata from the session."""
    session.pop("_manual_challenge_type", None)


def _translate_diagnostic(context, diagnostic: str | None) -> str | None:
    """Return a localized diagnosis string when a known diagnostic is available."""
    if not diagnostic:
        return None

    key = DIAGNOSTIC_I18N_KEYS.get(diagnostic)
    return tr(context, key) if key else diagnostic


def _format_artifact_note(context, artifacts: dict[str, str] | None) -> str:
    """Return a short text block that points to saved debug artifacts."""
    if not artifacts:
        return ""

    lines = ["", tr(context, "offer_debug_saved")]
    screenshot_path = artifacts.get("screenshot")
    html_path = artifacts.get("html")
    if screenshot_path:
        lines.append(tr(context, "offer_debug_screenshot", path=screenshot_path))
    if html_path:
        lines.append(tr(context, "offer_debug_html", path=html_path))
    return "\n".join(lines)


def _format_diagnostic_note(context, diagnostic: str | None) -> str:
    """Return a short diagnostic block for no-offer results."""
    if not diagnostic:
        return ""
    localized = _translate_diagnostic(context, diagnostic)
    return (
        f"\n\n{tr(context, 'offer_diagnosis_label')}: {localized}"
        if localized
        else ""
    )


def _diagnostic_means_embedded_trial(diagnostic: str | None) -> bool:
    """Return True when the page diagnosis indicates an eligible trial without a captured link."""
    normalized = (diagnostic or "").lower()
    return "embedded" in normalized and "trial" in normalized


def _is_manual_challenge_error(exc: Exception) -> bool:
    """Return True when login failed because Google requested non-TOTP verification."""
    message = str(exc).lower()
    return "no authenticator option found" in message and "requires" in message


def _is_dead_driver_error(exc: Exception) -> bool:
    """Return True when Chrome/WebDriver session has already died."""
    message = str(exc).lower()
    signals = (
        "invalid session id",
        "session deleted",
        "not connected to devtools",
        "chrome not reachable",
        "target window already closed",
        "disconnected",
    )
    return any(signal in message for signal in signals)


def _driver_session_is_alive(driver) -> bool:
    """Return False when the stored WebDriver can no longer answer commands."""
    if not driver:
        return False
    try:
        _ = driver.current_url
        return True
    except Exception:
        return False


def _looks_like_proxy_error(exc: Exception) -> bool:
    """Return True for errors that are likely caused by proxy/network transport."""
    message = str(exc).lower()
    signals = (
        "proxy",
        "timeout",
        "timed out",
        "connection",
        "tunnel",
        "refused",
        "dns",
        "net::err",
        "ssl",
        "unreachable",
    )
    return any(signal in message for signal in signals)


def _proxy_has_auth(proxy_url: str | None) -> bool:
    """Return True when the selected proxy carries username credentials."""
    if not proxy_url:
        return False
    try:
        return bool(parse_proxy_parts(proxy_url).get("username"))
    except Exception:
        return False


def _is_proxy_policy_error(exc: Exception) -> bool:
    """Return True when the upstream proxy provider blocks the destination by policy."""
    message = str(exc).lower()
    signals = (
        "bad_endpoint",
        "robots.txt",
        "policy_20130",
        "policy_20140",
        "blocked this target",
        "blocked this destination by policy",
    )
    return any(signal in message for signal in signals)


def _resolve_attempt_device(session: dict, attempt: int):
    """Return the device profile for this attempt and whether it is freshly minted."""
    route_tag = session.get("proxy") or "__direct__"
    needs_fresh_device = attempt > 1 and config.REGENERATE_DEVICE_ON_RETRY
    existing_device = session.get("device")
    existing_route_tag = session.get("_device_route_tag")

    if needs_fresh_device or not existing_device or existing_route_tag != route_tag:
        device = create_device_profile(
            network_identity=session.get("network_identity"),
            profile_name=session.get("device_profile"),
        )
        session["device"] = device
        session["_device_route_tag"] = route_tag
        return device, True

    return existing_device, False


async def _safe_send_bot_message(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    text: str,
    **kwargs,
) -> None:
    """Send a Telegram message without letting a transport timeout crash the handler."""
    try:
        await context.bot.send_message(chat_id=chat_id, text=text, **kwargs)
    except Exception as exc:
        logger.warning("Failed to send Telegram message to chat %s: %s", chat_id, exc)


async def _send_proxy_identity_panel(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    proxy_url: str | None,
    title: str,
    proxy_session_token: str | None = None,
) -> dict[str, str] | None:
    """Probe the selected proxy and send a readable info panel."""
    try:
        result = await asyncio.to_thread(
            inspect_connection,
            proxy_url,
            proxy_session_token,
        )
    except Exception as exc:
        logger.warning(
            "Failed to inspect proxy identity for chat %s via %s: %s",
            chat_id,
            mask_proxy_url(proxy_url),
            exc,
        )
        await _safe_send_bot_message(
            context,
            chat_id=chat_id,
            text=(
                f"{title}\n"
                f"🌐 Proxy: {mask_proxy_url(proxy_url)}\n"
                f"⚠️ Detailed proxy identity lookup failed: {exc}"
            ),
        )
        return None

    await _safe_send_bot_message(
        context,
        chat_id=chat_id,
        text=format_connection_identity(result, title=title),
    )
    return result


async def _report_offer(
    update_or_chat_id,
    context,
    session,
    offer_link,
    artifacts: dict[str, str] | None = None,
    diagnostic: str | None = None,
) -> None:
    """Send the offer result message."""
    chat_id = (
        update_or_chat_id
        if isinstance(update_or_chat_id, int)
        else update_or_chat_id.effective_chat.id
    )
    if offer_link:
        session["offer_link"] = offer_link
        text = tr(context, "offer_found_html", offer_link=offer_link)
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode="HTML",
                reply_markup=quick_actions_inline_keyboard(context),
            )
        except Exception:
            await context.bot.send_message(
                chat_id=chat_id,
                text=tr(context, "offer_found_plain", offer_link=offer_link),
                reply_markup=main_menu_keyboard(context),
            )
    else:
        message_key = (
            "offer_embedded_trial_detected"
            if _diagnostic_means_embedded_trial(diagnostic)
            else "offer_not_found_now"
        )
        await context.bot.send_message(
            chat_id=chat_id,
            text=tr(
                context,
                message_key,
                diagnostic_note=_format_diagnostic_note(context, diagnostic),
                artifact_note=_format_artifact_note(context, artifacts),
            ),
            reply_markup=quick_actions_inline_keyboard(context),
        )


async def check_offer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Run Google One automation and report the result."""
    max_offer_attempts = 3
    message = await prepare_action_message(update)
    chat_id = update.effective_chat.id
    session = get_session(chat_id)
    stale_driver = session.pop("_driver", None)
    close_driver(stale_driver)
    _clear_pending_verification(session)

    if not session.get("email") or not session.get("password"):
        await message.reply_text(
            tr(context, "offer_no_credentials"),
            reply_markup=main_menu_keyboard(context),
        )
        return ConversationHandler.END

    last_check = LAST_CHECK_TIME.get(chat_id, 0)
    elapsed = time.time() - last_check
    if elapsed < CHECK_OFFER_COOLDOWN:
        remaining = int(CHECK_OFFER_COOLDOWN - elapsed)
        mins, secs = divmod(remaining, 60)
        await message.reply_text(
            tr(context, "offer_cooldown_wait", mins=mins, secs=secs)
        )
        return ConversationHandler.END
    LAST_CHECK_TIME[chat_id] = time.time()

    if CHROME_SEMAPHORE.locked():
        await message.reply_text(
            tr(context, "offer_capacity_busy"),
            reply_markup=main_menu_keyboard(context),
        )
        LAST_CHECK_TIME.pop(chat_id, None)
        return ConversationHandler.END

    await message.reply_text(tr(context, "offer_starting_secure_check"))

    offer_link = None
    no_offer_artifacts: dict[str, str] | None = None
    no_offer_diagnostic: str | None = None
    try:
        async with CHROME_SEMAPHORE:
            email_str = bytes(session["email"]).decode("utf-8")
            pw_str = bytes(session["password"]).decode("utf-8")
            proxy_session_token = ensure_proxy_session_token(
                session,
                seed=f"{chat_id}{int(time.time())}",
            )
            used_proxies: set[str] = set()
            proxy_url = None
            fast_start_visible_auth_proxy = False
            if config.PROXY_ENABLED and not session.get("proxy_disabled"):
                proxy_url = PROXY_MANAGER.get_proxy(preferred=session.get("proxy"))
                if proxy_url:
                    session["proxy"] = proxy_url
                    fast_start_visible_auth_proxy = bool(
                        config.HEADLESS
                        and config.START_VISIBLE_WITH_AUTH_PROXY
                        and _proxy_has_auth(proxy_url)
                    )
                    if not fast_start_visible_auth_proxy:
                        identity = await _send_proxy_identity_panel(
                            context,
                            chat_id,
                            proxy_url,
                            tr(context, "proxy_panel_title"),
                            proxy_session_token,
                        )
                        if identity:
                            session["network_identity"] = identity
                        else:
                            session.pop("network_identity", None)
                    else:
                        logger.info(
                            "Skipping pre-launch proxy identity lookup for chat %s so Chrome can appear sooner on authenticated proxy startup.",
                            chat_id,
                        )
                else:
                    session.pop("proxy", None)
                    identity = await _send_proxy_identity_panel(
                        context,
                        chat_id,
                        None,
                        tr(context, "direct_panel_title"),
                        proxy_session_token,
                    )
                    if identity:
                        session["network_identity"] = identity
                    else:
                        session.pop("network_identity", None)
            else:
                session.pop("proxy", None)
                identity = await _send_proxy_identity_panel(
                    context,
                    chat_id,
                    None,
                    tr(context, "direct_panel_title"),
                    proxy_session_token,
                )
                if identity:
                    session["network_identity"] = identity
                else:
                    session.pop("network_identity", None)

            for attempt in range(1, max_offer_attempts + 1):
                device, fresh_device = _resolve_attempt_device(session, attempt)
                if attempt > 1:
                    retry_device_note = (
                        tr(context, "offer_retry_note_fresh")
                        if fresh_device
                        else tr(context, "offer_retry_note_same")
                    )
                    await message.reply_text(
                        tr(
                            context,
                            "offer_retry_attempt",
                            attempt=attempt,
                            max_attempts=max_offer_attempts,
                            note=retry_device_note,
                        )
                    )

                driver = None
                preserve_driver = False
                proxy_latency_ms = 0.0
                try:
                    should_skip_precheck = bool(
                        proxy_url
                        and attempt == 1
                        and fast_start_visible_auth_proxy
                    )
                    if proxy_url and config.PROXY_PRECHECK_ENABLED and not should_skip_precheck:
                        try:
                            probe_result = await asyncio.to_thread(
                                probe_google_signin,
                                proxy_url,
                                config.PROXY_PRECHECK_TIMEOUT_SECONDS,
                                proxy_session_token,
                            )
                            proxy_latency_ms = float(probe_result["latency_ms"])
                            logger.info(
                                "Proxy preflight passed for chat %s via %s in %.2f ms",
                                chat_id,
                                mask_proxy_url(proxy_url),
                                proxy_latency_ms,
                            )
                        except Exception as exc:
                            logger.warning(
                                "Proxy preflight failed for chat %s via %s: %s",
                                chat_id,
                                mask_proxy_url(proxy_url),
                                exc,
                            )
                            if attempt < max_offer_attempts and not _is_proxy_policy_error(exc):
                                PROXY_MANAGER.mark_failed(proxy_url, "precheck_failed")
                                used_proxies.add(proxy_url)
                                next_proxy = PROXY_MANAGER.get_proxy(excluded=used_proxies)
                                if next_proxy:
                                    proxy_url = next_proxy
                                    session["proxy"] = proxy_url
                                    await _safe_send_bot_message(
                                        context,
                                        chat_id=chat_id,
                                        text=tr(context, "offer_proxy_precheck_failed_rotate"),
                                    )
                                    identity = await _send_proxy_identity_panel(
                                        context,
                                        chat_id,
                                        proxy_url,
                                        tr(context, "proxy_rotated_title"),
                                        proxy_session_token,
                                    )
                                    if identity:
                                        session["network_identity"] = identity
                                    else:
                                        session.pop("network_identity", None)
                                    continue
                            raise GoogleAutomationError(
                                tr(
                                    context,
                                    "offer_proxy_precheck_failed_error",
                                    error=exc,
                                )
                            ) from exc
                    elif should_skip_precheck:
                        logger.info(
                            "Skipping proxy preflight for chat %s on first authenticated-proxy visible launch.",
                            chat_id,
                        )

                    try:
                        if (
                            attempt == 1
                            and config.HEADLESS
                            and config.START_VISIBLE_WITH_AUTH_PROXY
                            and _proxy_has_auth(proxy_url)
                        ):
                            await _safe_send_bot_message(
                                context,
                                chat_id=chat_id,
                                text=tr(context, "offer_opening_visible_browser"),
                            )
                        driver, status = await asyncio.to_thread(
                            start_login,
                            email_str,
                            pw_str,
                            device,
                            proxy_url=proxy_url,
                            proxy_session_token=proxy_session_token,
                        )
                    except GoogleAutomationError as exc:
                        if config.HEADLESS and _is_manual_challenge_error(exc):
                            await message.reply_text(
                                tr(context, "offer_manual_verification_reopen")
                            )
                            driver, status = await asyncio.to_thread(
                                start_login,
                                email_str,
                                pw_str,
                                device,
                                False,
                                proxy_url,
                                proxy_session_token,
                            )
                        else:
                            raise

                    if proxy_url:
                        PROXY_MANAGER.mark_success(proxy_url, latency_ms=proxy_latency_ms)

                    if status == "needs_totp":
                        totp_secret = session.get("totp_secret")
                        if totp_secret:
                            try:
                                import pyotp

                                code = pyotp.TOTP(totp_secret).now()
                                logger.info(
                                    "Auto-generated TOTP code for chat %s (attempt %d)",
                                    chat_id,
                                    attempt,
                                )
                                accepted = await asyncio.to_thread(submit_2fa_code, driver, code)
                                if not accepted:
                                    close_driver(driver)
                                    driver = None
                                    await message.reply_text(
                                        tr(context, "offer_auto_totp_rejected"),
                                        reply_markup=main_menu_keyboard(context),
                                    )
                                    return ConversationHandler.END

                                await message.reply_text(
                                    tr(
                                        context,
                                        "offer_login_success_checking",
                                        attempt=attempt,
                                        max_attempts=max_offer_attempts,
                                    )
                                )
                                offer_link = await asyncio.to_thread(check_offer_with_driver, driver)
                                if not offer_link and attempt == max_offer_attempts:
                                    no_offer_diagnostic = await asyncio.to_thread(
                                        diagnose_offer_page,
                                        driver,
                                    )
                                    no_offer_artifacts = await asyncio.to_thread(
                                        dump_offer_debug_artifacts,
                                        driver,
                                        chat_id,
                                        attempt,
                                        device.session_id,
                                    )
                            except Exception as exc:
                                logger.warning("Auto-TOTP failed: %s", exc)
                                close_driver(driver)
                                driver = None
                                await message.reply_text(
                                    tr(context, "offer_auto_totp_error", error=exc),
                                    reply_markup=main_menu_keyboard(context),
                                )
                                return ConversationHandler.END
                        else:
                            session["_driver"] = driver
                            preserve_driver = True
                            await _safe_send_bot_message(
                                context,
                                chat_id=chat_id,
                                text=tr(context, "offer_2fa_required"),
                                parse_mode="Markdown",
                            )
                            return AWAIT_2FA_CODE
                    elif status == "needs_manual_verification":
                        challenge_type = getattr(
                            driver,
                            "_autopixel_challenge_type",
                            "manual verification",
                        )
                        session["_driver"] = driver
                        session["_manual_challenge_type"] = challenge_type
                        preserve_driver = True
                        await _safe_send_bot_message(
                            context,
                            chat_id=chat_id,
                            text=tr(
                                context,
                                "offer_manual_required",
                                challenge_type=challenge_type,
                            ),
                            parse_mode="Markdown",
                        )
                        return AWAIT_MANUAL_VERIFICATION
                    else:
                        await message.reply_text(
                            tr(
                                context,
                                "offer_login_success_checking",
                                attempt=attempt,
                                max_attempts=max_offer_attempts,
                            )
                        )
                        offer_link = await asyncio.to_thread(check_offer_with_driver, driver)
                        if not offer_link and attempt == max_offer_attempts:
                            no_offer_diagnostic = await asyncio.to_thread(
                                diagnose_offer_page,
                                driver,
                            )
                            no_offer_artifacts = await asyncio.to_thread(
                                dump_offer_debug_artifacts,
                                driver,
                                chat_id,
                                attempt,
                                device.session_id,
                            )
                except GoogleAutomationError as exc:
                    if (
                        proxy_url
                        and _looks_like_proxy_error(exc)
                        and not _is_proxy_policy_error(exc)
                        and attempt < max_offer_attempts
                    ):
                        PROXY_MANAGER.mark_failed(proxy_url, "automation_error")
                        used_proxies.add(proxy_url)
                        next_proxy = PROXY_MANAGER.get_proxy(excluded=used_proxies)
                        if next_proxy:
                            proxy_url = next_proxy
                            session["proxy"] = proxy_url
                            await _safe_send_bot_message(
                                context,
                                chat_id=chat_id,
                                text=tr(context, "offer_proxy_transport_rotating"),
                            )
                            identity = await _send_proxy_identity_panel(
                                context,
                                chat_id,
                                proxy_url,
                                tr(context, "proxy_rotated_title"),
                                proxy_session_token,
                            )
                            if identity:
                                session["network_identity"] = identity
                            else:
                                session.pop("network_identity", None)
                            continue
                    raise
                except Exception as exc:
                    if (
                        proxy_url
                        and _looks_like_proxy_error(exc)
                        and not _is_proxy_policy_error(exc)
                        and attempt < max_offer_attempts
                    ):
                        PROXY_MANAGER.mark_failed(proxy_url, "runtime_error")
                        used_proxies.add(proxy_url)
                        next_proxy = PROXY_MANAGER.get_proxy(excluded=used_proxies)
                        if next_proxy:
                            proxy_url = next_proxy
                            session["proxy"] = proxy_url
                            await _safe_send_bot_message(
                                context,
                                chat_id=chat_id,
                                text=tr(context, "offer_runtime_network_rotating"),
                            )
                            identity = await _send_proxy_identity_panel(
                                context,
                                chat_id,
                                proxy_url,
                                tr(context, "proxy_rotated_title"),
                                proxy_session_token,
                            )
                            if identity:
                                session["network_identity"] = identity
                            else:
                                session.pop("network_identity", None)
                            continue
                    raise
                finally:
                    if driver and not preserve_driver:
                        close_driver(driver)

                if offer_link:
                    logger.info(
                        "Offer found on attempt %d for chat %s: %s",
                        attempt,
                        chat_id,
                        offer_link,
                    )
                    break

                logger.info(
                    "No offer found on attempt %d/%d for chat %s",
                    attempt,
                    max_offer_attempts,
                    chat_id,
                )

                if attempt < max_offer_attempts:
                    delay = random.randint(15, 30)
                    await message.reply_text(
                        tr(context, "offer_not_found_retrying", delay=delay)
                    )
                    await asyncio.sleep(delay)
                    next_retry_device = (
                        tr(context, "offer_retry_device_fresh")
                        if config.REGENERATE_DEVICE_ON_RETRY
                        else tr(context, "offer_retry_device_same")
                    )
                    await message.reply_text(
                        tr(
                            context,
                            "offer_starting_retry",
                            next_attempt=attempt + 1,
                            max_attempts=max_offer_attempts,
                            device_note=next_retry_device,
                        )
                    )

    except GoogleAutomationError as exc:
        await _safe_send_bot_message(
            context,
            chat_id=chat_id,
            text=tr(context, "offer_automation_error", error=exc),
            parse_mode="HTML",
            reply_markup=main_menu_keyboard(context),
        )
        return ConversationHandler.END
    except Exception as exc:
        logger.exception("Unexpected error in check_offer for chat %s", chat_id)
        await _safe_send_bot_message(
            context,
            chat_id=chat_id,
            text=tr(context, "offer_unexpected_error", error=exc),
        )
        return ConversationHandler.END
    finally:
        pw = session.get("password")
        if isinstance(pw, bytearray):
            secure_wipe(pw)
        session.pop("password", None)

    if not offer_link:
        await _safe_send_bot_message(
            context,
            chat_id=chat_id,
            text=tr(
                context,
                "offer_not_found_after_attempts",
                attempts=max_offer_attempts,
                diagnostic_note=_format_diagnostic_note(context, no_offer_diagnostic),
                artifact_note=_format_artifact_note(context, no_offer_artifacts),
            ),
            reply_markup=quick_actions_inline_keyboard(context),
        )
        return ConversationHandler.END

    await _report_offer(update, context, session, offer_link)
    return ConversationHandler.END


async def handle_2fa_code(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle the TOTP code submitted by the user during 2FA."""
    chat_id = update.effective_chat.id
    session = get_session(chat_id)
    code = update.message.text.strip()
    preserve_driver = False
    artifacts: dict[str, str] | None = None
    diagnostic: str | None = None

    try:
        await update.message.delete()
    except Exception:
        pass

    driver = session.pop("_driver", None)
    if not driver:
        _clear_pending_verification(session)
        await _safe_send_bot_message(
            context,
            chat_id=chat_id,
            text=tr(context, "offer_session_expired"),
            reply_markup=main_menu_keyboard(context),
        )
        return ConversationHandler.END

    if not _driver_session_is_alive(driver):
        _clear_pending_verification(session)
        close_driver(driver)
        await _safe_send_bot_message(
            context,
            chat_id=chat_id,
            text=tr(context, "offer_verification_session_closed"),
            reply_markup=main_menu_keyboard(context),
        )
        return ConversationHandler.END

    if not code.isdigit() or len(code) != 6:
        await _safe_send_bot_message(
            context,
            chat_id=chat_id,
            text=tr(context, "offer_invalid_code"),
            reply_markup=main_menu_keyboard(context),
        )
        session["_driver"] = driver
        return AWAIT_2FA_CODE

    await _safe_send_bot_message(
        context,
        chat_id=chat_id,
        text=tr(context, "offer_verifying_code"),
    )

    try:
        async with CHROME_SEMAPHORE:
            accepted = await asyncio.to_thread(submit_2fa_code, driver, code)

            if not accepted:
                session["_driver"] = driver
                preserve_driver = True
                await _safe_send_bot_message(
                    context,
                    chat_id=chat_id,
                    text=tr(context, "offer_code_rejected"),
                    reply_markup=main_menu_keyboard(context),
                )
                return AWAIT_2FA_CODE

            try:
                offer_link = await asyncio.to_thread(check_offer_with_driver, driver)
                if not offer_link:
                    diagnostic = await asyncio.to_thread(diagnose_offer_page, driver)
                    device = session.get("device")
                    artifacts = await asyncio.to_thread(
                        dump_offer_debug_artifacts,
                        driver,
                        chat_id,
                        None,
                        getattr(device, "session_id", None),
                    )
            finally:
                close_driver(driver)

    except Exception as exc:
        logger.exception("Error in 2FA for chat %s", chat_id)
        close_driver(driver)
        error_text = (
            tr(context, "offer_verification_session_closed")
            if _is_dead_driver_error(exc)
            else tr(context, "offer_generic_error", error=exc)
        )
        await _safe_send_bot_message(context, chat_id=chat_id, text=error_text)
        return ConversationHandler.END
    finally:
        if not preserve_driver:
            _clear_pending_verification(session)
            pw = session.get("password")
            if isinstance(pw, bytearray):
                secure_wipe(pw)
            session.pop("password", None)

    await _report_offer(
        chat_id,
        context,
        session,
        offer_link,
        artifacts=artifacts,
        diagnostic=diagnostic,
    )
    return ConversationHandler.END


async def handle_manual_verification(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> int:
    """Continue the offer flow after the user finishes manual Chrome verification."""
    chat_id = update.effective_chat.id
    session = get_session(chat_id)
    preserve_driver = False
    artifacts: dict[str, str] | None = None
    diagnostic: str | None = None

    try:
        await update.message.delete()
    except Exception:
        pass

    driver = session.pop("_driver", None)
    challenge_type = session.get("_manual_challenge_type", "manual verification")
    if not driver:
        _clear_pending_verification(session)
        await _safe_send_bot_message(
            context,
            chat_id=chat_id,
            text=tr(context, "offer_session_expired"),
            reply_markup=main_menu_keyboard(context),
        )
        return ConversationHandler.END

    if not _driver_session_is_alive(driver):
        _clear_pending_verification(session)
        close_driver(driver)
        await _safe_send_bot_message(
            context,
            chat_id=chat_id,
            text=tr(context, "offer_verification_window_closed"),
            reply_markup=main_menu_keyboard(context),
        )
        return ConversationHandler.END

    await _safe_send_bot_message(
        context,
        chat_id=chat_id,
        text=tr(context, "offer_checking_chrome_window"),
    )

    try:
        state = await asyncio.to_thread(resolve_manual_login, driver, 8)
        if state == "needs_totp":
            session["_driver"] = driver
            preserve_driver = True
            await _safe_send_bot_message(
                context,
                chat_id=chat_id,
                text=tr(context, "offer_2fa_required_after_manual"),
                parse_mode="Markdown",
                reply_markup=main_menu_keyboard(context),
            )
            return AWAIT_2FA_CODE

        if state in {"challenge", "signin"}:
            session["_driver"] = driver
            preserve_driver = True
            await _safe_send_bot_message(
                context,
                chat_id=chat_id,
                text=tr(
                    context,
                    "offer_google_waiting_verification",
                    challenge_type=challenge_type,
                ),
                parse_mode="Markdown",
                reply_markup=main_menu_keyboard(context),
            )
            return AWAIT_MANUAL_VERIFICATION

        await _safe_send_bot_message(
            context,
            chat_id=chat_id,
            text=tr(context, "offer_verification_completed_checking"),
        )
        try:
            offer_link = await asyncio.to_thread(check_offer_with_driver, driver)
            if not offer_link:
                diagnostic = await asyncio.to_thread(diagnose_offer_page, driver)
                device = session.get("device")
                artifacts = await asyncio.to_thread(
                    dump_offer_debug_artifacts,
                    driver,
                    chat_id,
                    None,
                    getattr(device, "session_id", None),
                )
        finally:
            close_driver(driver)

    except Exception as exc:
        logger.exception("Error in manual verification for chat %s", chat_id)
        close_driver(driver)
        error_text = (
            tr(context, "offer_verification_window_closed")
            if _is_dead_driver_error(exc)
            else tr(context, "offer_generic_error", error=exc)
        )
        await _safe_send_bot_message(context, chat_id=chat_id, text=error_text)
        return ConversationHandler.END
    finally:
        if not preserve_driver:
            _clear_pending_verification(session)
            pw = session.get("password")
            if isinstance(pw, bytearray):
                secure_wipe(pw)
            session.pop("password", None)

    await _report_offer(
        chat_id,
        context,
        session,
        offer_link,
        artifacts=artifacts,
        diagnostic=diagnostic,
    )
    return ConversationHandler.END


async def cancel_2fa(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancel a pending verification step and close the driver."""
    chat_id = update.effective_chat.id
    session = get_session(chat_id)
    driver = session.pop("_driver", None)
    _clear_pending_verification(session)
    close_driver(driver)
    await update.message.reply_text(
        tr(context, "offer_verification_cancelled"),
        reply_markup=main_menu_keyboard(context),
    )
    return ConversationHandler.END


async def offer_timeout(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle conversation timeout by cleaning up a pending verification flow."""
    if update and update.effective_chat:
        chat_id = update.effective_chat.id
        session = SESSION_STORE.get(chat_id, {})
        driver = session.pop("_driver", None)
        _clear_pending_verification(session)
        close_driver(driver)
        await context.bot.send_message(
            chat_id=chat_id,
            text=tr(context, "offer_verification_timed_out"),
            reply_markup=main_menu_keyboard(context),
        )
    return ConversationHandler.END
