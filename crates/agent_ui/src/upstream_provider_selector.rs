use std::rc::Rc;
use std::sync::Arc;

use acp_thread::{
    AgentModelId, AgentModelSelector, AgentUpstreamProvider, AgentUpstreamProviderPricing,
};
use fuzzy_nucleo::StringMatchCandidate;
use gpui::{AnyView, Context, DismissEvent, Entity, Subscription, Task, Window, prelude::*};
use picker::{Picker, PickerDelegate, popover_menu::PickerPopoverMenu};
use ui::{
    Button, Chip, IconName, ListItem, ListItemSpacing, PopoverMenuHandle, Tooltip, prelude::*,
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
    tags: Vec<SharedString>,
    pricing: Option<AgentUpstreamProviderPricing>,
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
            tags: Vec::new(),
            pricing: None,
            delta: None,
            meta: None,
            is_selected: false,
            is_focused: false,
        }
    }

    fn tags(mut self, tags: Vec<SharedString>) -> Self {
        self.tags = tags;
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
    fn render(self, _window: &mut Window, _cx: &mut App) -> impl IntoElement {
        let discount = self.pricing.as_ref().and_then(|pricing| pricing.discount);
        let price_column = self
            .pricing
            .as_ref()
            .filter(|pricing| pricing.has_prices())
            .map(|pricing| match self.delta.as_ref() {
                Some(delta) => v_flex()
                    .id("upstream-provider-price-deltas")
                    .items_end()
                    .children(delta_labels(pricing, delta))
                    .tooltip(price_tooltip(pricing))
                    .into_any_element(),
                None => v_flex()
                    .id("upstream-provider-prices")
                    .items_end()
                    .children(price_labels(pricing))
                    .tooltip(price_tooltip(pricing))
                    .into_any_element(),
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
                            .child(Label::new(self.title).truncate())
                            .children(self.tags.into_iter().map(Chip::new)),
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

/// Shown when either price is unknown, so the columns stay aligned.
const UNKNOWN_DELTA: &str = "\u{2013}";

/// One label per price, input first.
fn price_labels(pricing: &AgentUpstreamProviderPricing) -> impl Iterator<Item = Label> + '_ {
    let color = if pricing.discount.is_some_and(|discount| discount > 0.0) {
        Color::Success
    } else {
        Color::Muted
    };

    [pricing.input, pricing.output, pricing.cache_read]
        .into_iter()
        .filter_map(move |price| {
            Some(
                Label::new(format_price(price?))
                    .size(LabelSize::XSmall)
                    .color(color),
            )
        })
}

/// One label per price difference to the selected provider, input first.
fn delta_labels<'a>(
    pricing: &'a AgentUpstreamProviderPricing,
    delta: &'a AgentUpstreamProviderPricing,
) -> impl Iterator<Item = Label> + 'a {
    [
        (pricing.input, delta.input),
        (pricing.output, delta.output),
        (pricing.cache_read, delta.cache_read),
    ]
    .into_iter()
    .filter_map(move |(price, difference)| {
        // Fields the provider does not price have no line.
        price?;
        Some(
            match difference {
                // One of the prices is unknown.
                None => Label::new(UNKNOWN_DELTA).color(Color::Muted),
                // Below the displayed precision the two prices are the same.
                Some(difference) if (difference.abs() * 1_000_000.0).round() < 1.0 => {
                    Label::new(format_price(0.0)).color(Color::Muted)
                }
                Some(difference) if difference > 0.0 => {
                    Label::new(format!("+{}", format_price(difference.abs()))).color(Color::Error)
                }
                Some(difference) => {
                    Label::new(format!("-{}", format_price(difference.abs()))).color(Color::Success)
                }
            }
            .size(LabelSize::XSmall),
        )
    })
}

/// A tooltip that labels every price on its own line, input first.
fn price_tooltip(
    pricing: &AgentUpstreamProviderPricing,
) -> impl Fn(&mut Window, &mut App) -> AnyView + 'static {
    let lines: Vec<(&'static str, SharedString)> = [
        ("Input", pricing.input),
        ("Output", pricing.output),
        ("Cache read", pricing.cache_read),
    ]
    .into_iter()
    .filter_map(|(label, price)| Some((label, format_price(price?).into())))
    .collect();
    Tooltip::element(move |_, _| {
        v_flex()
            .gap_1()
            .min_w(rems(9.))
            .children(lines.clone().into_iter().map(|(label, price)| {
                h_flex()
                    .gap_3()
                    .justify_between()
                    .child(Label::new(label).size(LabelSize::Small).color(Color::Muted))
                    .child(Label::new(price).size(LabelSize::Small))
            }))
            .into_any_element()
    })
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
                    (
                        StringMatchCandidate::new(index, provider.name.clone()),
                        StringMatchCandidate::new(index, provider.tags.join(" ")),
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
            .tags(provider.tags.clone())
            .is_selected(is_selected)
            .is_focused(selected);
        if let Some(pricing) = provider.pricing.clone() {
            let delta = if is_selected {
                None // no delta for the currently selected element
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
