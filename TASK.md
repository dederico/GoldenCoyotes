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

## Tareas Completadas - 2026-03-06

### Feedback Implementado (Retroalimentación 4)

#### 1. Filtro de Oportunidades por Red de Contactos ✅
**Estado:** COMPLETADO
**Descripción:** Solo visualizar oportunidades que estén dentro de MI red de contactos
**Implementación:**
- Modificado `get_opportunities()` en `database_setup.py` para aceptar parámetro `network_only`
- Query SQL con JOIN a tabla `connections` para filtrar por red (líneas 375-387)
- Actualizado `get_public_opportunities()` para usar filtro de red (golden_coyotes_after_feedback.py:574)
- Actualizado `get_directed_opportunities()` para filtrar por tags (líneas 576-588)
- Actualizado `opportunities_status` para combinar red + mis oportunidades (líneas 527-555)
**Impacto crítico:** Los usuarios ahora SOLO ven oportunidades de personas con las que están conectados

#### 2. Endpoints REST para Gestión de Conexiones ✅
**Estado:** COMPLETADO
**Descripción:** Crear endpoints propios para conexiones transparentes
**Implementación:**
- `GET /api/connections` - Obtener mis conexiones (con filtro por status)
- `GET /api/connections/pending` - Ver solicitudes pendientes
- `POST /api/connections/request` - Solicitar nueva conexión
- `POST /api/connections/accept` - Aceptar solicitud de conexión
- `POST /api/connections/reject` - Rechazar solicitud de conexión
**Archivos modificados:**
- `golden_coyotes_after_feedback.py` (líneas 397-513)
**Seguridad:** Validación de permisos - solo puedes aceptar/rechazar tus propias solicitudes

#### 3. Métodos de Base de Datos para Conexiones ✅
**Estado:** COMPLETADO
**Descripción:** Gestión transparente de conexiones con métodos dedicados
**Implementación:**
- `get_user_connections(user_id, status)` - Obtener conexiones por estado
- `get_pending_connection_requests(user_id)` - Solicitudes pendientes recibidas
- `accept_connection(connection_id, user_id)` - Aceptar conexión
- `reject_connection(connection_id, user_id)` - Rechazar conexión
- `is_connected(user_id, target_user_id)` - Verificar si dos usuarios están conectados
**Archivos modificados:**
- `database_setup.py` (líneas 433-581)

#### 4. Vista de Mis Contactos ✅
**Estado:** COMPLETADO
**Descripción:** Interfaz para visualizar y gestionar contactos y solicitudes
**Implementación:**
- Nueva ruta `/mis-contactos` con vista completa
- Métricas: Total de contactos y solicitudes pendientes
- Tabla de contactos con información completa
- Tarjetas de solicitudes pendientes con botones de aceptar/rechazar
- JavaScript para gestionar acciones AJAX
**Archivos modificados:**
- `golden_coyotes_after_feedback.py` (líneas 558-576, 2834-3040)
**Template:** MIS_CONTACTOS_TEMPLATE con Bootstrap 5

#### 5. Scroll Mejorado de Contactos ✅
**Estado:** YA COMPLETADO (Retro 2)
**Descripción:** Contenedor con scroll para seleccionar contactos
**Ubicación:** Formulario "Oportunidad Dirigida" (golden_coyotes_after_feedback.py:1359-1374)
- max-height: 300px
- overflow-y: auto
- Grid de 2 columnas responsive

#### 6. Suite de Tests Completa ✅
**Estado:** COMPLETADO
**Descripción:** Tests unitarios y de integración para retro 4
**Implementación:**
- 12 tests en total (todos pasando)
- `TestRetro4NetworkFiltering` - Tests de filtro de red
- `TestRetro4ConnectionManagement` - Tests de gestión de conexiones
- `TestRetro4Integration` - Tests de integración de flujo completo
**Archivo:** `tests/test_retro4_network_filtering.py`
**Resultado:** 12/12 tests pasando exitosamente

#### 7. Configuración PostgreSQL para Persistencia ✅
**Estado:** DOCUMENTADO
**Descripción:** Soporte para PostgreSQL en Render para persistencia real
**Implementación:**
- Documentación completa en `RENDER_POSTGRESQL_SETUP.md`
- Instrucciones paso a paso para configurar PostgreSQL en Render
- Solución al problema de datos no persistentes en SQLite
**Archivos creados:**
- `RENDER_POSTGRESQL_SETUP.md`

### Detalle de Implementación Técnica

**Cambios en Base de Datos:**
- Método `get_opportunities()` ahora soporta `network_only=True` y `requesting_user_id`
- Query SQL usa INNER JOIN con tabla `connections` para filtrar por red
- Verifica que `status = 'accepted'` en conexiones
- Excluye oportunidades propias del usuario (`o.user_id != ?`)

**Cambios en Backend:**
- 5 nuevos endpoints REST para gestión de conexiones
- 5 nuevos métodos de base de datos para conexiones transparentes
- Actualización de todos los métodos que llaman a `get_opportunities()`
- Seguridad: validación de permisos en endpoints

**Cambios en Frontend:**
- Nueva vista `/mis-contactos` con gestión completa
- JavaScript AJAX para aceptar/rechazar conexiones
- Template con métricas, solicitudes pendientes y tabla de contactos

**Testing:**
- Suite completa de 12 tests unitarios y de integración
- Cobertura de filtro de red, gestión de conexiones y flujos completos
- Todos los tests pasando exitosamente

### Archivos Modificados

**Archivos principales:**
1. `database_setup.py` - Métodos de conexiones y filtro de red
2. `golden_coyotes_after_feedback.py` - Endpoints REST y vista de contactos
3. `tests/test_retro4_network_filtering.py` (nuevo) - Suite de tests
4. `RENDER_POSTGRESQL_SETUP.md` (nuevo) - Documentación de PostgreSQL

**Líneas de código:**
- `database_setup.py`: ~200 líneas nuevas/modificadas
- `golden_coyotes_after_feedback.py`: ~350 líneas nuevas/modificadas
- `test_retro4_network_filtering.py`: ~400 líneas nuevas

### Pendiente de Deployment

**Antes de deployar a Render:**
- [ ] Configurar PostgreSQL en Render (seguir RENDER_POSTGRESQL_SETUP.md)
- [ ] Agregar variable de entorno `DATABASE_URL`
- [ ] Agregar `psycopg2-binary` a `requirements.txt`
- [ ] Ejecutar `/admin/seed-demo-data` UNA VEZ después del primer deploy

**Testing en Producción:**
1. Verificar que filtro de red funciona (solo veo oportunidades de mi red)
2. Probar solicitar conexión con otro usuario
3. Probar aceptar/rechazar solicitudes
4. Verificar que después de conectar, veo oportunidades del nuevo contacto
5. Verificar persistencia de datos después de un nuevo deploy

---

**Última actualización:** 2026-03-06
**Por:** Claude Code Assistant
