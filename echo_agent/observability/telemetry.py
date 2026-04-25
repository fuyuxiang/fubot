"""OpenTelemetry integration — TracerProvider, MeterProvider, GenAI conventions."""

from __future__ import annotations

from typing import Any

from loguru import logger

try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource
    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False


class TelemetryManager:
    """Manages OpenTelemetry providers and exporters.

    Gracefully degrades when opentelemetry is not installed.
    """

    def __init__(
        self,
        service_name: str = "echo-agent",
        otel_endpoint: str = "",
        export_interval_ms: int = 5000,
    ):
        self._service_name = service_name
        self._endpoint = otel_endpoint
        self._export_interval = export_interval_ms
        self._tracer: Any = None
        self._meter: Any = None
        self._initialized = False

    @property
    def available(self) -> bool:
        return _HAS_OTEL

    def setup(self) -> None:
        if not _HAS_OTEL:
            logger.info("OpenTelemetry not installed — telemetry disabled")
            return

        resource = Resource.create({"service.name": self._service_name})

        # Tracer
        tracer_provider = TracerProvider(resource=resource)
        if self._endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                exporter = OTLPSpanExporter(endpoint=self._endpoint)
                tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
            except ImportError:
                tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        else:
            tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(tracer_provider)
        self._tracer = trace.get_tracer("echo-agent", "0.1.0")

        # Meter
        meter_provider = MeterProvider(resource=resource)
        metrics.set_meter_provider(meter_provider)
        self._meter = metrics.get_meter("echo-agent", "0.1.0")

        self._initialized = True
        logger.info("OpenTelemetry initialized: service={}, endpoint={}", self._service_name, self._endpoint or "console")

    def get_tracer(self) -> Any:
        return self._tracer

    def get_meter(self) -> Any:
        return self._meter

    def shutdown(self) -> None:
        if not _HAS_OTEL or not self._initialized:
            return
        try:
            provider = trace.get_tracer_provider()
            if hasattr(provider, "shutdown"):
                provider.shutdown()
        except Exception as e:
            logger.warning("OTel shutdown error: {}", e)
