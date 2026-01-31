# Task List - Golden Coyotes Platform

## Tareas Completadas - 2026-01-30

### Feedback Implementado (Junio 2025)

#### 1. Confirmación de Oportunidad Enviada ✅
**Estado:** COMPLETADO
**Descripción:** Agregar notificación que informe/confirme que la oportunidad fue subida exitosamente
**Implementación:**
- Mensaje flash de confirmación en ambas rutas (subir-oportunidad y oportunidad-dirigida)
- Redirección automática a "/mis-oportunidades" después de crear la oportunidad
- Mensaje: "¡Oportunidad publicada exitosamente! Podrás verla en 'Mis Oportunidades'."
**Archivos modificados:**
- `golden_coyotes_after_feedback.py` (líneas 171, 206)

#### 2. Visualizar Oportunidades Enviadas ✅
**Estado:** COMPLETADO
**Descripción:** Crear área donde el usuario pueda visualizar las oportunidades que ha enviado
**Implementación:**
- Nueva ruta `/mis-oportunidades`
- Template HTML dedicado (MIS_OPORTUNIDADES_TEMPLATE)
- Vista de tarjetas con todas las oportunidades publicadas por el usuario
- Información mostrada: título, descripción, tipo, industria, vigencia, fecha de creación
**Archivos modificados:**
- `golden_coyotes_after_feedback.py` (líneas 379-385, 2423-2505)

#### 3. Ver Lista de Oportunidades con Estado ✅
**Estado:** COMPLETADO
**Descripción:** Crear ruta para ver el estado de todas las oportunidades
**Implementación:**
- Nueva ruta `/opportunities-status`
- Template HTML dedicado (OPPORTUNITIES_STATUS_TEMPLATE)
- Dashboard con estadísticas: Activas, Por Vencer, Vencidas
- Tabla detallada con información de vigencia
- Indicadores visuales de estado (verde/amarillo/rojo)
**Archivos modificados:**
- `golden_coyotes_after_feedback.py` (líneas 387-406, 2507-2626)

#### 4. Vigencia de Oportunidades ✅
**Estado:** COMPLETADO
**Descripción:** Agregar atributo de vigencia a las oportunidades
**Implementación:**
- Campo `expiration_date` agregado a la tabla `opportunities` en base de datos
- Campo incluido en formularios de creación (subir-oportunidad y oportunidad-dirigida)
- Valor por defecto: 30 días desde la creación
- Validación automática de vigencia en vista de estado
- Cálculo automático de días restantes
- Clasificación: Activa (>7 días), Por Vencer (0-7 días), Vencida (<0 días)
**Archivos modificados:**
- `database_setup.py` (línea 71, líneas 298-305)
- `golden_coyotes_after_feedback.py` (líneas 163, 195, 441-454, 1149-1157, 1294-1302, 387-406)

### Detalle de Implementación Técnica

**Base de Datos:**
- Tabla `opportunities` actualizada con campo `expiration_date DATE`
- Método `create_opportunity` actualizado para soportar vigencia

**Backend:**
- Lógica de validación de vigencia automática
- Cálculo de días restantes
- Clasificación por estado de vigencia

**Frontend:**
- Formularios actualizados con campo de fecha de vigencia
- Input type="date" con validación min/max
- Templates HTML profesionales con Bootstrap 5
- Indicadores visuales de estado (badges de colores)
- Dashboard de métricas

### Pregunta Pendiente del Feedback

**¿Cada cuánto validar que la información está vigente o actualizada?**

**Opciones sugeridas:**
1. **Validación en tiempo real:** Cada vez que un usuario accede a la lista de oportunidades
2. **Tarea programada diaria:** Cron job que marque oportunidades vencidas como inactivas
3. **Notificaciones proactivas:** Enviar email 7 días antes del vencimiento
4. **Combinación:** Validación en tiempo real + notificaciones + tarea de limpieza semanal

**Recomendación:**
Implementar validación en tiempo real (ya implementada) + agregar tarea programada nocturna que:
- Marque oportunidades vencidas como `is_active = 0`
- Envíe notificaciones a usuarios con oportunidades por vencer en 7 días
- Limpie oportunidades vencidas hace más de 90 días

---

## Tareas Futuras (Sugerencias)

### Alta Prioridad
- [ ] Implementar tarea programada para desactivar oportunidades vencidas
- [ ] Agregar notificaciones por email antes del vencimiento
- [ ] Permitir renovar/extender vigencia de oportunidades
- [ ] Agregar filtros por estado de vigencia en búsqueda

### Media Prioridad
- [ ] Exportar oportunidades a CSV/Excel
- [ ] Estadísticas avanzadas de oportunidades por usuario
- [ ] Gráficos de tendencias de oportunidades

### Baja Prioridad
- [ ] Historial de cambios en oportunidades
- [ ] Versiones de oportunidades
- [ ] Templates de oportunidades frecuentes

---

**Última actualización:** 2026-01-30
**Por:** Claude Code Assistant
