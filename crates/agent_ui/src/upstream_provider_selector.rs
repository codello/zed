use std::rc::Rc;
use std::sync::Arc;

use acp_thread::{
    AgentModelId, AgentModelSelector, AgentUpstreamProvider, AgentUpstreamProviderPricing,
};
use fuzzy_nucleo::StringMatchCandidate;
use gpui::{
    AnyView, Context, DismissEvent, Entity, Subscription, Task, Window, img, prelude::*, rgba,
};
use picker::{Picker, PickerDelegate, popover_menu::PickerPopoverMenu};
use ui::{
    Button, Chip, Divider, IconName, ListItem, ListItemSpacing, PopoverMenuHandle, Tooltip,
    prelude::*,
};

use crate::model_search;

pub type UpstreamProviderPicker = Picker<UpstreamProviderPickerDelegate>;

pub fn upstream_provider_picker(
    selector: Rc<dyn AgentModelSelector>,
    window: &mut Window,
    cx: &mut Context<UpstreamProviderPicker>,
) -> UpstreamProviderPicker {
    let delegate = UpstreamProviderPickerDelegate::new(selector, window, cx);
    Picker::list(delegate, window, cx)
        .show_scrollbar(true)
        .initial_width(rems(24.))
}

#[derive(IntoElement)]
struct UpstreamProviderListItem {
    index: usize,
    title: SharedString,
    service_tier: Option<SharedString>,
    quantization: Option<SharedString>,
    region: Option<SharedString>,
    headquarters: Option<SharedString>,
    datacenters: Vec<SharedString>,
    icon_url: Option<SharedString>,
    pricing: Option<AgentUpstreamProviderPricing>,
    /// The difference to the selected provider's pricing.
    delta: Option<AgentUpstreamProviderPricing>,
    meta: Option<SharedString>,
    is_selected: bool,
    is_focused: bool,
}

impl UpstreamProviderListItem {
    fn new(index: usize, title: impl Into<SharedString>) -> Self {
        Self {
            index,
            title: title.into(),
            service_tier: None,
            quantization: None,
            region: None,
            headquarters: None,
            datacenters: Vec::new(),
            icon_url: None,
            pricing: None,
            delta: None,
            meta: None,
            is_selected: false,
            is_focused: false,
        }
    }

    fn service_tier(mut self, service_tier: Option<SharedString>) -> Self {
        self.service_tier = service_tier;
        self
    }

    fn quantization(mut self, quantization: Option<SharedString>) -> Self {
        self.quantization = quantization;
        self
    }

    fn region(mut self, region: Option<SharedString>) -> Self {
        self.region = region;
        self
    }

    fn headquarters(mut self, headquarters: Option<SharedString>) -> Self {
        self.headquarters = headquarters;
        self
    }

    fn datacenters(mut self, datacenters: Vec<SharedString>) -> Self {
        self.datacenters = datacenters;
        self
    }

    fn icon_url(mut self, icon_url: Option<SharedString>) -> Self {
        self.icon_url = icon_url;
        self
    }

    fn pricing(
        mut self,
        pricing: AgentUpstreamProviderPricing,
        delta: Option<AgentUpstreamProviderPricing>,
    ) -> Self {
        self.pricing = Some(pricing);
        self.delta = delta;
        self
    }

    fn meta(mut self, meta: impl Into<SharedString>) -> Self {
        self.meta = Some(meta.into());
        self
    }

    fn is_selected(mut self, is_selected: bool) -> Self {
        self.is_selected = is_selected;
        self
    }

    fn is_focused(mut self, is_focused: bool) -> Self {
        self.is_focused = is_focused;
        self
    }
}

impl RenderOnce for UpstreamProviderListItem {
    fn render(self, _window: &mut Window, cx: &mut App) -> impl IntoElement {
        let discount = self.pricing.as_ref().and_then(|pricing| pricing.discount);
        let mut name_detail_lines: Vec<(&'static str, SharedString)> = Vec::new();
        if let Some(headquarters) = &self.headquarters {
            name_detail_lines.push((
                "Headquarters",
                format!("{} {}", region_label(headquarters), headquarters).into(),
            ));
        }
        if !self.datacenters.is_empty() {
            let datacenters = self
                .datacenters
                .iter()
                .map(|datacenter| format!("{} {}", region_label(datacenter), datacenter))
                .collect::<Vec<_>>()
                .join(", ");
            name_detail_lines.push(("Datacenters", datacenters.into()));
        }
        let price_column = self.pricing.as_ref().and_then(|pricing| {
            let price = display_price(pricing)?;
            // Rows other than the selected one show the price delta
            let label = if let Some(delta) = self.delta.as_ref() {
                delta_label(delta)
            } else {
                // Discounted prices are highlighted only while no provider
                // is selected; the selected provider's price stays regular.
                let color = if !self.is_selected
                    && pricing.discount.is_some_and(|discount| discount > 0.0)
                {
                    Color::Success
                } else {
                    Color::Muted
                };
                Label::new(format_price(price))
                    .size(LabelSize::Small)
                    .color(color)
            };
            let column = v_flex()
                .id("upstream-provider-price")
                .items_end()
                .child(label);
            Some(column.tooltip(price_tooltip(pricing)).into_any_element())
        });

        ListItem::new(self.index)
            .inset(true)
            .spacing(ListItemSpacing::Sparse)
            .toggle_state(self.is_focused)
            .child(
                v_flex()
                    .w_full()
                    .gap_0p5()
                    .child(
                        h_flex()
                            .w_full()
                            .gap_1p5()
                            .when_some(self.icon_url, |this, url| {
                                this.child(
                                    img(url.as_ref())
                                        .size(px(14.))
                                        .rounded_sm()
                                        .bg(cx.theme().colors().element_disabled)
                                        .with_fallback(|| {
                                            h_flex()
                                                .size_full()
                                                .justify_center()
                                                .child(
                                                    Icon::new(IconName::Person)
                                                        .color(Color::Muted)
                                                        .size(IconSize::XSmall),
                                                )
                                                .into_any_element()
                                        }),
                                )
                            })
                            .child(
                                div()
                                    .id("provider-name")
                                    .overflow_hidden()
                                    .child(Label::new(self.title).truncate())
                                    .when(!name_detail_lines.is_empty(), |this| {
                                        this.tooltip(name_tooltip(name_detail_lines))
                                    }),
                            )
                            .when_some(self.region, |this, region| {
                                this.child(Label::new(region_label(&region)))
                            })
                            .when_some(self.service_tier, |this, tier| {
                                // The violet of the OpenRouter website's
                                // service tier chip; a brand color, not a
                                // theme color.
                                this.child(
                                    Chip::new(tier)
                                        .label_color(Color::Custom(rgba(0xc4b5fdff).into()))
                                        .bg_color(rgba(0x8b5cf640).into())
                                        .border_color(rgba(0x8b5cf640).into()),
                                )
                            })
                            .when_some(self.quantization, |this, quantization| {
                                this.child(Chip::new(quantization))
                            }),
                    )
                    .when_some(self.meta, |this, meta| {
                        this.child(Label::new(meta).size(LabelSize::XSmall).color(Color::Muted))
                    }),
            )
            .end_slot(
                h_flex()
                    .pr_2()
                    .gap_3()
                    .when_some(format_discount(discount), |this, discount| {
                        this.child(Chip::new(discount).label_color(Color::Success))
                    })
                    .when_some(price_column, |this, column| this.child(column))
                    .when(self.is_selected, |this| {
                        this.child(Icon::new(IconName::Check).color(Color::Accent))
                    }),
            )
    }
}

/// Shown when the reference price is unknown.
const UNKNOWN_DELTA: &str = "\u{2013}";

/// The price shown in the row: the blended estimate when available, falling
/// back to the effective and then the list input price.
fn display_price(pricing: &AgentUpstreamProviderPricing) -> Option<f64> {
    pricing
        .blended
        .or(pricing.effective_input)
        .or(pricing.input)
}

/// The row's price difference to the selected provider, or a dash when it
/// is unknown.
fn delta_label(delta: &AgentUpstreamProviderPricing) -> Label {
    let label = match display_price(delta) {
        None => Label::new(UNKNOWN_DELTA).color(Color::Muted),
        Some(delta) if (delta * 1_000_000.0).round() == 0.0 => {
            // Below the displayed precision the two prices are the same.
            Label::new(format_price(0.0)).color(Color::Muted)
        }
        Some(delta) if delta > 0.0 => {
            Label::new(format!("+{}", format_price(delta))).color(Color::Error)
        }
        Some(delta) => Label::new(format!("-{}", format_price(-delta))).color(Color::Success),
    };
    label.size(LabelSize::Small)
}

/// A tooltip that shows details about the company operating the endpoint.
fn name_tooltip(
    lines: Vec<(&'static str, SharedString)>,
) -> impl Fn(&mut Window, &mut App) -> AnyView + 'static {
    Tooltip::element(move |_, _| {
        v_flex()
            .gap_1()
            .min_w(rems(9.))
            .children(lines.clone().into_iter().map(|(label, value)| {
                h_flex()
                    .gap_3()
                    .justify_between()
                    .child(Label::new(label).size(LabelSize::Small).color(Color::Muted))
                    .child(Label::new(value).size(LabelSize::Small))
            }))
            .into_any_element()
    })
}

/// A tooltip that shows the list prices the displayed price is built from,
/// and the effective input price with the cache hit rate behind it.
fn price_tooltip(
    pricing: &AgentUpstreamProviderPricing,
) -> impl Fn(&mut Window, &mut App) -> AnyView + 'static {
    let mut rows: Vec<(&'static str, SharedString)> = Vec::new();
    for (label, price) in [
        ("Input/M", pricing.input),
        ("Output/M", pricing.output),
        ("Cache read/M", pricing.cache_read),
        ("Cache write/M", pricing.cache_write),
    ] {
        if let Some(price) = price {
            rows.push((label, format_price(price).into()));
        }
    }
    let effective_line =
        pricing
            .cache_hit_rate
            .zip(pricing.effective_input)
            .map(|(rate, effective_input)| {
                (
                    format!("Effective input at {:.1}% cache rate", rate * 100.0),
                    format_price(effective_input),
                )
            });
    let notes: Vec<&'static str> = if pricing.cache_read.is_none() {
        vec!["No prompt caching on this endpoint"]
    } else if pricing.cache_hit_rate.is_none() {
        vec!["No global cache data; showing the list input price"]
    } else {
        Vec::new()
    };
    Tooltip::element(move |_, _| {
        v_flex()
            .gap_1()
            .min_w(rems(9.))
            .children(rows.clone().into_iter().map(|(label, value)| {
                h_flex()
                    .gap_3()
                    .justify_between()
                    .child(Label::new(label).size(LabelSize::Small).color(Color::Muted))
                    .child(Label::new(value).size(LabelSize::Small))
            }))
            .child(Divider::horizontal())
            .children(effective_line.clone().map(|(label, value)| {
                h_flex()
                    .gap_3()
                    .justify_between()
                    .child(
                        Label::new(label)
                            .size(LabelSize::XSmall)
                            .color(Color::Muted),
                    )
                    .child(
                        Label::new(value)
                            .size(LabelSize::XSmall)
                            .color(Color::Muted),
                    )
            }))
            .children(notes.iter().map(|note| {
                Label::new(*note)
                    .size(LabelSize::XSmall)
                    .color(Color::Muted)
            }))
            .into_any_element()
    })
}

/// The flag emoji of a two-letter region code, for example `eu`. Falls back
/// to the region code itself when there is no flag for it.
fn region_label(region: &str) -> SharedString {
    let indicator =
        |letter: char| char::from_u32('🇦' as u32 + letter.to_ascii_uppercase() as u32 - 'A' as u32);
    let mut chars = region.chars();
    if let (Some(first), Some(second), None) = (chars.next(), chars.next(), chars.next()) {
        if first.is_ascii_alphabetic() && second.is_ascii_alphabetic() {
            if let (Some(first), Some(second)) = (indicator(first), indicator(second)) {
                return format!("{first}{second}").into();
            }
        }
    }
    region.into()
}

/// The whole percentage taken off the list price, if it rounds to at least one
/// percent.
fn format_discount(discount: Option<f64>) -> Option<SharedString> {
    // Round first so the guard and the label agree on the displayed value.
    let percent = (discount? * 100.0).round();
    (percent >= 1.0).then(|| format!("{percent:.0}% off").into())
}

/// Formats a price per million tokens in USD.
fn format_price(price: f64) -> String {
    let formatted = format!("{price:.6}");
    let (whole, fraction) = formatted.split_once('.').unwrap_or((&formatted, ""));
    let fraction = fraction.trim_end_matches('0');
    if fraction.len() >= 2 {
        format!("${whole}.{fraction}")
    } else {
        format!("${whole}.{fraction:0<2}")
    }
}

fn format_meta(provider: &AgentUpstreamProvider) -> Option<SharedString> {
    let mut parts = Vec::new();
    if let Some(throughput) = provider.throughput {
        parts.push(if throughput >= 10.0 {
            format!("{throughput:.0} t/s")
        } else {
            format!("{throughput:.1} t/s")
        });
    }
    if let Some(latency) = provider.latency {
        parts.push(if latency < 1000.0 {
            format!("{latency:.0}ms TTFT")
        } else {
            format!("{:.1}s TTFT", latency / 1000.0)
        });
    }
    (!parts.is_empty()).then(|| parts.join(" · ").into())
}

pub struct UpstreamProviderPickerDelegate {
    selector: Rc<dyn AgentModelSelector>,
    model_id: Option<AgentModelId>,
    providers: Option<Vec<AgentUpstreamProvider>>,
    selected: Option<String>,
    filtered_entries: Vec<AgentUpstreamProvider>,
    selected_index: usize,
    _refresh_task: Task<()>,
}

impl UpstreamProviderPickerDelegate {
    fn new(
        selector: Rc<dyn AgentModelSelector>,
        window: &mut Window,
        cx: &mut Context<UpstreamProviderPicker>,
    ) -> Self {
        let mut this = Self {
            selector,
            model_id: None,
            providers: None,
            selected: None,
            filtered_entries: Vec::new(),
            selected_index: 0,
            _refresh_task: Task::ready(()),
        };
        this.refresh(window, cx);
        this
    }

    pub fn refresh(&mut self, window: &mut Window, cx: &mut Context<UpstreamProviderPicker>) {
        let selector = self.selector.clone();
        self._refresh_task = cx.spawn_in(window, async move |this, cx| {
            let Ok(selected_model_task) = cx.update(|_, cx| selector.selected_model(cx)) else {
                return;
            };
            let selected_model = selected_model_task.await.ok();
            let model_id = selected_model.as_ref().map(|model| model.id.clone());

            let unchanged = this
                .read_with(cx, |this, _| this.delegate.model_id == model_id)
                .unwrap_or(false);
            if unchanged {
                return;
            }

            let (providers, selected) = if let Some(model_id) = &model_id {
                let Ok(providers_task) =
                    cx.update(|_, cx| selector.upstream_providers(model_id, cx))
                else {
                    return;
                };
                let Ok(selected_task) = cx.update(|_, cx| selector.selected_upstream_provider(cx))
                else {
                    return;
                };
                let (providers, selected) = futures::join!(providers_task, selected_task);
                let providers = providers
                    .map_err(|error| log::warn!("Failed to list upstream providers: {error:#}"))
                    .ok()
                    .flatten();
                let selected = selected
                    .map_err(|error| {
                        log::warn!("Failed to read the selected upstream provider: {error:#}")
                    })
                    .ok()
                    .flatten();
                (providers, selected)
            } else {
                (None, None)
            };

            this.update_in(cx, |this, window, cx| {
                this.delegate.model_id = model_id;
                this.delegate.providers = providers;
                this.delegate.selected = selected;
                // Rebuild the matches from the new providers instead of fetching them again.
                this.refresh(window, cx);
            })
            .ok();
        });
    }

    pub fn is_visible(&self) -> bool {
        self.providers.is_some()
    }

    /// The selected provider, if it is still listed.
    fn selected_provider(&self) -> Option<&AgentUpstreamProvider> {
        let selected = self.selected.as_deref()?;
        self.providers
            .as_ref()?
            .iter()
            .find(|provider| provider.id == selected)
    }

    fn selected_index_for_selection(&self) -> usize {
        let Some(selected) = &self.selected else {
            return 0;
        };
        self.filtered_entries
            .iter()
            .position(|provider| &provider.id == selected)
            .unwrap_or(0)
    }
}

impl PickerDelegate for UpstreamProviderPickerDelegate {
    type ListItem = AnyElement;

    fn name() -> &'static str {
        "upstream provider selector"
    }

    fn match_count(&self) -> usize {
        self.filtered_entries.len()
    }

    fn selected_index(&self) -> usize {
        self.selected_index
    }

    fn set_selected_index(&mut self, ix: usize, _: &mut Window, cx: &mut Context<Picker<Self>>) {
        self.selected_index = ix.min(self.filtered_entries.len().saturating_sub(1));
        cx.notify();
    }

    fn can_select(&self, ix: usize, _window: &mut Window, _cx: &mut Context<Picker<Self>>) -> bool {
        self.filtered_entries.get(ix).is_some()
    }

    fn placeholder_text(&self, _window: &mut Window, _cx: &mut App) -> Arc<str> {
        "Select a provider…".into()
    }

    fn update_matches(
        &mut self,
        query: String,
        window: &mut Window,
        cx: &mut Context<Picker<Self>>,
    ) -> Task<()> {
        let providers = self.providers.clone().unwrap_or_default();

        cx.spawn_in(window, async move |this, cx| {
            let filtered = model_search::search(
                providers,
                &query,
                cx.background_executor().clone(),
                |index, provider| {
                    let labels = [
                        provider.service_tier.as_deref(),
                        provider.quantization.as_deref(),
                        provider.region.as_deref(),
                    ]
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>()
                    .join(" ");
                    (
                        StringMatchCandidate::new(index, provider.name.clone()),
                        StringMatchCandidate::new(index, labels),
                    )
                },
            )
            .await;

            this.update_in(cx, |this, window, cx| {
                this.delegate.filtered_entries = filtered;
                let new_index = this.delegate.selected_index_for_selection();
                this.set_selected_index(new_index, Some(picker::Direction::Down), true, window, cx);
                cx.notify();
            })
            .ok();
        })
    }

    fn confirm(&mut self, _secondary: bool, _window: &mut Window, cx: &mut Context<Picker<Self>>) {
        let Some(provider) = self.filtered_entries.get(self.selected_index) else {
            return;
        };
        // Choosing the current provider again returns to automatic routing.
        let provider =
            (self.selected.as_deref() != Some(provider.id.as_str())).then(|| provider.id.clone());
        self.selector
            .select_upstream_provider(provider.clone(), cx)
            .detach_and_log_err(cx);
        self.selected = provider;
        cx.emit(DismissEvent);
    }

    fn dismissed(&mut self, window: &mut Window, cx: &mut Context<Picker<Self>>) {
        cx.defer_in(window, |picker, window, cx| {
            picker.set_query("", window, cx);
        });
    }

    fn render_match(
        &self,
        ix: usize,
        selected: bool,
        _: &mut Window,
        _cx: &mut Context<Picker<Self>>,
    ) -> Option<Self::ListItem> {
        let provider = self.filtered_entries.get(ix)?;
        let is_selected = self.selected.as_deref() == Some(provider.id.as_str());
        let mut item = UpstreamProviderListItem::new(ix, provider.name.clone())
            .service_tier(provider.service_tier.clone())
            .quantization(provider.quantization.clone())
            .region(provider.region.clone())
            .headquarters(provider.headquarters.clone())
            .datacenters(provider.datacenters.clone())
            .icon_url(provider.icon_url.clone())
            .is_selected(is_selected)
            .is_focused(selected);
        if let Some(pricing) = provider.pricing.clone() {
            let delta = if is_selected {
                None
            } else {
                self.selected_provider()
                    .and_then(|provider| provider.pricing.as_ref())
                    .map(|reference| pricing.difference(reference))
            };
            item = item.pricing(pricing, delta);
        }
        if let Some(meta) = format_meta(provider) {
            item = item.meta(meta);
        }
        Some(item.into_any_element())
    }
}

pub struct UpstreamProviderSelector {
    picker: Entity<UpstreamProviderPicker>,
    menu_handle: PopoverMenuHandle<UpstreamProviderPicker>,
    _subscription: Subscription,
}

impl UpstreamProviderSelector {
    pub fn new(
        selector: Rc<dyn AgentModelSelector>,
        menu_handle: PopoverMenuHandle<UpstreamProviderPicker>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let picker = cx.new(move |cx| upstream_provider_picker(selector, window, cx));
        let subscription = cx.observe(&picker, |_, _, cx| cx.notify());
        Self {
            picker,
            menu_handle,
            _subscription: subscription,
        }
    }

    pub fn refresh(&self, window: &mut Window, cx: &mut Context<Self>) {
        self.picker.update(cx, |picker, cx| {
            picker.delegate.refresh(window, cx);
        });
    }
}

impl Render for UpstreamProviderSelector {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let picker = self.picker.read(cx);
        if !picker.delegate.is_visible() {
            return div().into_any_element();
        }

        let selected_label = picker
            .delegate
            .selected_provider()
            .map(|provider| provider.name.clone());
        let is_automatic = selected_label.is_none();
        let label = selected_label.unwrap_or_else(|| SharedString::from("Any Provider"));
        let (color, icon) = if self.menu_handle.is_deployed() {
            (Color::Muted, IconName::ChevronUp)
        } else if is_automatic {
            (Color::Placeholder, IconName::ChevronDown)
        } else {
            (Color::Muted, IconName::ChevronDown)
        };

        PickerPopoverMenu::new(
            self.picker.clone(),
            Button::new("upstream-provider", label)
                .label_size(LabelSize::Small)
                .color(color)
                .selected_label_color(Color::Accent)
                .end_icon(Icon::new(icon).color(color).size(IconSize::XSmall)),
            Tooltip::text("Choose the upstream provider"),
            gpui::Anchor::BottomRight,
            cx,
        )
        .with_handle(self.menu_handle.clone())
        .render(window, cx)
        .into_any_element()
    }
}

#[cfg(test)]
mod tests {
    use super::{format_discount, format_price};

    #[test]
    fn format_price_keeps_two_decimals_and_drops_the_rest() {
        assert_eq!(format_price(30.0), "$30.00");
        assert_eq!(format_price(0.1), "$0.10");
        assert_eq!(format_price(0.06), "$0.06");
        assert_eq!(format_price(0.0), "$0.00");
    }

    #[test]
    fn format_price_keeps_significant_decimals() {
        assert_eq!(format_price(0.006), "$0.006");
        assert_eq!(format_price(0.0015), "$0.0015");
        assert_eq!(format_price(0.000015), "$0.000015");
        assert_eq!(format_price(0.125), "$0.125");
    }

    #[test]
    fn format_discount_hides_rounding_noise() {
        assert_eq!(format_discount(None), None);
        assert_eq!(format_discount(Some(0.0)), None);
        assert_eq!(format_discount(Some(0.004)), None);
    }

    #[test]
    fn format_discount_rounds_to_whole_percent() {
        assert_eq!(format_discount(Some(0.25)).as_deref(), Some("25% off"));
        assert_eq!(format_discount(Some(0.8)).as_deref(), Some("80% off"));
        assert_eq!(format_discount(Some(0.125)).as_deref(), Some("13% off"));
    }
}
