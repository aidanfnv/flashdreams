// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/* Catalog-driven command-line model client over the packet-driver TCP adapter. */
#include <conio.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cJSON.h"
#include "lwip/dhcp.h"
#include "lwip/init.h"
#include "lwip/sys.h"
#include "lwip/stats.h"
#include "lwip/tcp.h"
#include "lwip/timeouts.h"
#include "netif/ethernet.h"
#include "packet.h"
#include "mp4.h"
#include "image.h"
#include "vbe.h"

enum { MAX_HEADER = 16384, MAX_PAYLOAD = 12 * 1024 * 1024 };
static struct netif interface;
static struct tcp_pcb *stream;
static int connected, stopped, failed;
static unsigned interrupt = 0x60;
static uint32_t request_id;
static unsigned char prefix[12], *rgb;
static char header[MAX_HEADER + 1];
static unsigned prefix_used, header_size, header_used, payload_size, payload_used;
static int waiting, complete;
static u32_t last_progress;
static unsigned timeout_seconds;
static u32_t run_started, run_limit_ms;
static int run_expired;
static unsigned rgb_capacity;
static int graphics_active, transfer_reported;
static u32_t last_transfer, last_output;
static int consume_status(void);

static uint32_t network_word(const unsigned char *bytes)
{
    return ((uint32_t)bytes[0] << 24) | ((uint32_t)bytes[1] << 16) |
           ((uint32_t)bytes[2] << 8) | bytes[3];
}

static void put_network_word(unsigned char *bytes, uint32_t value)
{
    for (unsigned i = 0; i < 4; ++i)
        bytes[i] = value >> (24 - i * 8);
}

static void network_error(const char *message)
{
    fprintf(stderr, "%s\n", message);
    failed = 1;
}

static int poll_network(void)
{
    packet_poll(&interface);
    sys_check_timeouts();
    while (kbhit()) {
        if (getch() == 27)
            stopped = 1;
    }
    if (run_limit_ms && (u32_t)(sys_now() - run_started) >= run_limit_ms) {
        run_expired = stopped = 1;
    }
    return failed || stopped ? -1 : 0;
}

static int wait_progress(void)
{
    if (poll_network())
        return -1;
    if (timeout_seconds && (u32_t)(sys_now() - last_progress) >= timeout_seconds * 1000u) {
        fprintf(stderr, "Request %lu: prefix=%u header=%u/%u payload=%u/%u packets=%lu rejected=%u\n",
                (unsigned long)request_id, prefix_used, header_used, header_size,
                payload_used, payload_size, packet_received(), packet_dropped());
        if (stream)
            fprintf(stderr, "TCP state=%u receive_window=%lu next_sequence=%lu send_buffer=%u\n",
                    stream->state, (unsigned long)stream->rcv_wnd,
                    (unsigned long)stream->rcv_nxt, tcp_sndbuf(stream));
        fprintf(stderr, "TCP received=%u dropped=%u checksum_errors=%u memory_errors=%u\n",
                lwip_stats.tcp.recv, lwip_stats.tcp.drop,
                lwip_stats.tcp.chkerr, lwip_stats.tcp.memerr);
        packet_trace_save();
        fprintf(stderr, "Network operation timed out after %u seconds without progress.\n", timeout_seconds);
        failed = 1;
        return -1;
    }
    return 0;
}

static void tcp_error(void *argument, err_t error)
{
    (void)argument;
    stream = NULL; /* lwIP has already freed the PCB before this callback. */
    fprintf(stderr, "TCP connection failed: %d\n", error);
    failed = 1;
}

static err_t tcp_connected(void *argument, struct tcp_pcb *pcb, err_t error)
{
    (void)argument;
    (void)pcb;
    if (error != ERR_OK)
        return error;
    connected = 1;
    last_progress = sys_now();
    return ERR_OK;
}

static err_t tcp_received(void *argument, struct tcp_pcb *pcb,
                          struct pbuf *packet, err_t error)
{
    unsigned offset = 0;
    (void)argument;
    (void)error;
    if (!packet) {
        network_error("Model service closed the connection.");
        return ERR_OK;
    }
    while (offset < packet->tot_len) {
        unsigned available = packet->tot_len - offset, count;
        void *destination;
        if (!waiting || complete)
            goto invalid;
        if (prefix_used < sizeof(prefix)) {
            count = sizeof(prefix) - prefix_used;
            if (count > available)
                count = available;
            pbuf_copy_partial(packet, prefix + prefix_used, count, offset);
            prefix_used += count;
            offset += count;
            if (prefix_used == sizeof(prefix)) {
                header_size = network_word(prefix + 4);
                payload_size = network_word(prefix + 8);
                if (memcmp(prefix, "FDM1", 4) || !header_size ||
                    header_size > MAX_HEADER || payload_size > MAX_PAYLOAD)
                    goto invalid;
                if (payload_size > rgb_capacity) {
                    unsigned char *larger = realloc(rgb, payload_size);
                    if (!larger) goto invalid;
                    rgb = larger;
                    rgb_capacity = payload_size;
                }
            }
            continue;
        }
        if (header_used < header_size) {
            count = header_size - header_used;
            destination = header + header_used;
        } else {
            count = payload_size - payload_used;
            destination = rgb + payload_used;
        }
        if (count > available)
            count = available;
        pbuf_copy_partial(packet, destination, count, offset);
        if (header_used < header_size)
            header_used += count;
        else
            payload_used += count;
        offset += count;
        if (header_used == header_size && payload_used == payload_size) {
            header[header_size] = '\0';
            int status = consume_status();
            if (status < 0) goto invalid;
            if (status) {
                prefix_used = header_used = payload_used = 0;
                header_size = payload_size = 0;
            } else complete = 1;
        }
    }
    last_progress = sys_now();
    tcp_recved(pcb, packet->tot_len);
    pbuf_free(packet);
    return ERR_OK;
invalid:
    network_error("Invalid or unsolicited model-service message.");
    pbuf_free(packet);
    tcp_abort(pcb);
    stream = NULL;
    return ERR_ABRT;
}

static int send_bytes(const void *data, unsigned size)
{
    const unsigned char *bytes = data;
    while (size) {
        unsigned count = tcp_sndbuf(stream);
        if (count > size)
            count = size;
        if (count) {
            err_t result = tcp_write(stream, bytes, count, TCP_WRITE_FLAG_COPY);
            if (result == ERR_OK) {
                bytes += count;
                size -= count;
                last_progress = sys_now();
                tcp_output(stream);
            } else if (result != ERR_MEM) {
                network_error("Cannot queue TCP request.");
                return -1;
            }
        }
        if (wait_progress())
            return -1;
    }
    return 0;
}

static int json_finite(const cJSON *value)
{
    if (cJSON_IsNumber(value) && !isfinite(value->valuedouble))
        return 0;
    for (const cJSON *child = value->child; child; child = child->next)
        if (!json_finite(child))
            return 0;
    return 1;
}

static int field_number(const cJSON *object, const char *key, uint32_t expected)
{
    const cJSON *value = cJSON_GetObjectItemCaseSensitive(object, key);
    return cJSON_IsNumber(value) && value->valuedouble == expected;
}

static int field_string(const cJSON *object, const char *key, const char *expected)
{
    const cJSON *value = cJSON_GetObjectItemCaseSensitive(object, key);
    return cJSON_IsString(value) && !strcmp(value->valuestring, expected);
}

static int consume_status(void)
{
    cJSON *message = cJSON_ParseWithLengthOpts(header, header_size + 1, NULL, 1);
    if (!field_string(message, "type", "status")) {
        cJSON_Delete(message);
        return 0;
    }
    const cJSON *phase = cJSON_GetObjectItemCaseSensitive(message, "phase");
    if (payload_size || !json_finite(message) ||
        !field_number(message, "id", request_id) || !cJSON_IsString(phase)) {
        cJSON_Delete(message);
        return -1;
    }
    int valid = 1;
    if (!strcmp(phase->valuestring, "loading")) {
        if (!graphics_active) puts("Loading model...");
    } else if (!strcmp(phase->valuestring, "reusing")) {
        if (!graphics_active) puts("Reusing resident model.");
    } else if (!strcmp(phase->valuestring, "preparing")) {
        if (!graphics_active) puts("Preparing prompt and generation state...");
    } else if (!strcmp(phase->valuestring, "generating")) {
        const cJSON *block = cJSON_GetObjectItemCaseSensitive(message, "block");
        const cJSON *total = cJSON_GetObjectItemCaseSensitive(message, "total_blocks");
        if (!cJSON_IsNumber(block) || block->valuedouble < 1 ||
            floor(block->valuedouble) != block->valuedouble) valid = 0;
        else if (!graphics_active) {
            printf("Generating block %.0f", block->valuedouble);
            if (cJSON_IsNumber(total)) printf("/%.0f", total->valuedouble);
            puts("...");
        }
        transfer_reported = 0;
    } else if (!strcmp(phase->valuestring, "transferring")) {
        if (!graphics_active && (!transfer_reported || (u32_t)(sys_now() - last_transfer) >= 1000)) {
            puts("Transferring RGB frames...");
            last_transfer = sys_now();
            transfer_reported = 1;
        }
    } else valid = 0;
    if (!graphics_active) fflush(stdout);
    cJSON_Delete(message);
    return valid ? 1 : -1;
}

static cJSON *request(const char *operation)
{
    cJSON *result = cJSON_CreateObject();
    if (!result || request_id == UINT32_MAX) {
        cJSON_Delete(result);
        network_error("Cannot allocate another request.");
        return NULL;
    }
    cJSON_AddNumberToObject(result, "id", ++request_id);
    cJSON_AddStringToObject(result, "op", operation);
    return result;
}

static cJSON *exchange_payload(cJSON *message, const char *type,
                               const void *payload, unsigned length)
{
    unsigned char outgoing[12] = {'F', 'D', 'M', '1'};
    cJSON *reply = NULL;
    char *text = message ? cJSON_PrintUnformatted(message) : NULL;
    cJSON_Delete(message);
    if (!text) {
        network_error("Cannot encode request.");
        return NULL;
    }
    unsigned size = strlen(text);
    if (!size || size > MAX_HEADER) {
        network_error("Request header exceeds protocol limits.");
        free(text);
        return NULL;
    }
    prefix_used = header_used = payload_used = 0;
    header_size = payload_size = 0;
    waiting = 1;
    complete = 0;
    last_progress = sys_now();
    put_network_word(outgoing + 4, size);
    put_network_word(outgoing + 8, length);
    if (send_bytes(outgoing, sizeof(outgoing)) || send_bytes(text, size) ||
        (length && send_bytes(payload, length)))
        goto done;
    while (!complete)
        if (wait_progress())
            goto done;
    reply = cJSON_ParseWithLengthOpts(header, header_size + 1, NULL, 1);
    int finished = !strcmp(type, "frames") && field_string(reply, "type", "finished");
    if (!cJSON_IsObject(reply) || !json_finite(reply) ||
        !field_number(reply, "id", request_id) ||
        (!field_string(reply, "type", type) && !finished)) {
        const cJSON *detail = cJSON_GetObjectItemCaseSensitive(reply, "message");
        fprintf(stderr, "Invalid response to request %lu%s%s\n", (unsigned long)request_id,
                cJSON_IsString(detail) ? ": " : "", cJSON_IsString(detail) ? detail->valuestring : "");
        failed = 1;
        cJSON_Delete(reply);
        reply = NULL;
    } else if ((strcmp(type, "frames") || finished) && payload_size) {
        network_error("Unexpected response payload.");
        cJSON_Delete(reply);
        reply = NULL;
    }
done:
    waiting = 0;
    free(text);
    return reply;
}

static cJSON *exchange(cJSON *message, const char *type)
{
    return exchange_payload(message, type, NULL, 0);
}

static void disconnect_stream(void)
{
    if (stream) {
        tcp_err(stream, NULL);
        tcp_recv(stream, NULL);
        tcp_abort(stream);
        stream = NULL;
    }
    connected = waiting = complete = 0;
}

static int connect_service(const char *host, unsigned port)
{
    ip_addr_t peer;
    cJSON *message, *reply;
    disconnect_stream();
    request_id = 0;
    prefix_used = header_used = payload_used = 0;
    header_size = payload_size = 0;
    if (!ipaddr_aton(host, &peer) || !(stream = tcp_new())) {
        network_error("Cannot allocate TCP connection or parse server IPv4 address.");
        return -1;
    }
    tcp_err(stream, tcp_error);
    tcp_recv(stream, tcp_received);
    last_progress = sys_now();
    if (tcp_connect(stream, &peer, port, tcp_connected) != ERR_OK) {
        network_error("Cannot begin TCP connection.");
        return -1;
    }
    while (!connected)
        if (wait_progress()) return -1;
    printf("Connected to %s:%u using lwIP " LWIP_VERSION_STRING "\n", host, port);
    message = request("hello");
    cJSON_AddStringToObject(message, "protocol", "flashdreams-model-service");
    cJSON_AddNumberToObject(message, "version", 1);
    cJSON_AddBoolToObject(message, "status_messages", 1);
    reply = exchange(message, "hello");
    if (!reply) return -1;
    int valid = field_string(reply, "protocol", "flashdreams-model-service") &&
                field_number(reply, "version", 1) && field_string(reply, "frame_format", "rgb24");
    int devices = cJSON_IsTrue(cJSON_GetObjectItemCaseSensitive(reply, "device_selection"));
    cJSON_Delete(reply);
    if (!valid) {
        network_error("Incompatible model-service handshake.");
        return -1;
    }
    return devices;
}

static err_t init_interface(struct netif *netif)
{
    return packet_open(netif, interrupt) ? ERR_IF : ERR_OK;
}

static int number(const char *text, unsigned long minimum, unsigned long maximum,
                   unsigned long *value)
{
    char *end;
    if (!*text || *text == '-')
        return -1;
    errno = 0;
    *value = strtoul(text, &end, 0);
    return errno || *end || *value < minimum || *value > maximum ? -1 : 0;
}

struct server_endpoint {
    const char *name, *host;
    unsigned port;
};

static int server_endpoint(char *text, struct server_endpoint *server)
{
    char *equals = strchr(text, '='), *colon = strrchr(text, ':');
    unsigned long port;
    ip_addr_t peer;
    if (!equals || !colon || colon <= equals + 1 || !text[0]) return -1;
    *equals = *colon = 0;
    unsigned length = strlen(text);
    if (!length || length > 63 || !((text[0] >= 'A' && text[0] <= 'Z') ||
                                  (text[0] >= 'a' && text[0] <= 'z'))) return -1;
    for (const char *c = text; *c; ++c)
        if (!((*c >= 'A' && *c <= 'Z') || (*c >= 'a' && *c <= 'z') ||
              (*c >= '0' && *c <= '9') || *c == '_' || *c == '-')) return -1;
    for (const char *c = colon + 1; *c; ++c)
        if (*c < '0' || *c > '9') return -1;
    if (!ipaddr_aton(equals + 1, &peer) || !colon[1]) return -1;
    /* Endpoint ports are decimal, including those with leading zeroes. */
    errno = 0;
    port = strtoul(colon + 1, NULL, 10);
    if (errno || !port || port > 65535) return -1;
    server->name = text;
    server->host = equals + 1;
    server->port = port;
    return 0;
}

static void usage(void)
{
    puts("Usage: FLSHDRMS remote [APPLICATION] [runtime options] -- [application options]\n"
         "  --list-models           List the remote catalog without opening a model\n"
         "  APPLICATION -- --help Show application options from the catalog\n"
         "  --pixel-width N --pixel-height N --fps N   Override generated video\n"
         "  --timeout SECONDS|unbound   Optional application runtime limit\n"
         "  --host IPv4 --port N   Service address (default 10.0.2.2:9390)\n"
         "  --server NAME=IPv4:PORT   Repeat for named servers; replaces --host/--port\n"
         "  --list-devices        Discover server GPUs without loading a model\n"
         "  --device ID|NAME/ID   Choose an inference device (qualify with multiple servers)\n"
         "  --ip IPv4 --netmask IPv4 --gateway IPv4   Static IP; omit --ip for DHCP\n"
         "  --packet-int 0x60 --bpp 24|32 --network-timeout SECONDS\n"
         "  --mode native-window|mp4   Local output (default native-window)\n"
         "  --output-path FILE     Required for mp4; encoded and saved on DOS\n"
         "  --error-path FILE      Optional stderr log for unattended runs\n"
         "  --check               Process four frames for verification\n"
         "  --check-frames N      Diagnostic check with N frames instead\n"
         "Application options after --:\n"
         "  --prompt TEXT | --prompt-file FILE --seed N --total-blocks N\n"
         "  --compile | --no-compile   Override the adapter's compilation default\n"
         "  --image-path FILE      PNG/JPEG decoded and resized on DOS\n"
         "  --initial-rgb FILE     Alternative: raw RGB24 at the model's frame size\n"
         "Supported options depend on the selected application's catalog.\n"
         "MP4 ends with generation; native-window stays open until Escape.\n"
         "Normal runs have no frame cap or default inactivity timeout.");
}

static unsigned char *load_file(const char *path, unsigned limit, unsigned *length)
{
    FILE *file = fopen(path, "rb");
    unsigned char *data = malloc(limit + 1);
    if (!file || !data) {
        if (file) fclose(file);
        free(data);
        fprintf(stderr, "Cannot load %s\n", path);
        return NULL;
    }
    *length = fread(data, 1, limit, file);
    int valid = fgetc(file) == EOF && !ferror(file);
    fclose(file);
    if (!valid) {
        fprintf(stderr, "%s exceeds %u bytes or could not be read.\n", path, limit);
        free(data);
        return NULL;
    }
    data[*length] = 0;
    return data;
}

static void describe_settings(const cJSON *schema, const char *label, int runtime)
{
    printf("%s:\n", label);
    const cJSON *spec;
    cJSON_ArrayForEach(spec, schema) {
        int frame_option = !strcmp(spec->string, "width") ||
                           !strcmp(spec->string, "height") || !strcmp(spec->string, "fps");
        if (frame_option != runtime) continue;
        printf("  --%s", frame_option && strcmp(spec->string, "fps") ? "pixel-" : "");
        for (const char *name = spec->string; *name; ++name)
            putchar(*name == '_' ? '-' : *name);
        const cJSON *type = cJSON_GetObjectItemCaseSensitive(spec, "type");
        if (!strcmp(spec->string, "compile")) printf(" | --no-compile");
        else printf(" %s", cJSON_IsString(type) ? type->valuestring : "value");
        const char *fields[] = {"default", "minimum", "maximum", "max_length"};
        for (unsigned i = 0; i < sizeof(fields) / sizeof(fields[0]); ++i) {
            const cJSON *value = cJSON_GetObjectItemCaseSensitive(spec, fields[i]);
            char *text = value ? cJSON_PrintUnformatted(value) : NULL;
            if (text) printf(" %s=%s", fields[i], text);
            free(text);
        }
        putchar('\n');
    }
}

static unsigned image_dimension(const cJSON *parameters, const cJSON *model, const char *key)
{
    const cJSON *value = cJSON_GetObjectItemCaseSensitive(parameters, key);
    if (!value) value = cJSON_GetObjectItemCaseSensitive(
        cJSON_GetObjectItemCaseSensitive(
            cJSON_GetObjectItemCaseSensitive(model, "parameters"), key), "default");
    if (!cJSON_IsNumber(value) || value->valuedouble < 1 || value->valuedouble > 2048 ||
        floor(value->valuedouble) != value->valuedouble) return 0;
    return value->valuedouble;
}

static int setting(cJSON *parameters, cJSON *inputs, const cJSON *model,
                   const char *option, const char *text)
{
    char key[128];
    cJSON *destination = parameters, *value = NULL;
    if (strncmp(option, "--", 2) || !option[2] || strlen(option + 2) >= sizeof(key))
        goto invalid;
    strcpy(key, option + 2);
    for (char *letter = key; *letter; ++letter)
        if (*letter == '-') *letter = '_';
    const cJSON *spec = cJSON_GetObjectItemCaseSensitive(
        cJSON_GetObjectItemCaseSensitive(model, "parameters"), key);
    if (!spec) {
        destination = inputs;
        spec = cJSON_GetObjectItemCaseSensitive(
            cJSON_GetObjectItemCaseSensitive(model, "inputs"), key);
    }
    if (!cJSON_IsObject(spec) || cJSON_GetObjectItemCaseSensitive(destination, key))
        goto invalid;
    if (field_string(spec, "type", "string")) {
        const cJSON *maximum = cJSON_GetObjectItemCaseSensitive(spec, "max_length");
        unsigned characters = 0;
        for (const unsigned char *letter = (const unsigned char *)text; *letter; ++letter)
            if ((*letter & 0xc0) != 0x80) ++characters;
        if (cJSON_IsNumber(maximum) && characters > maximum->valuedouble) goto invalid;
        value = cJSON_CreateString(text);
    } else if (field_string(spec, "type", "integer") || field_string(spec, "type", "number")) {
        char *end;
        errno = 0;
        double numeric = field_string(spec, "type", "integer")
                         ? strtoll(text, &end, 10) : strtod(text, &end);
        if (!*text || *end || errno || !isfinite(numeric) ||
            (field_string(spec, "type", "integer") &&
             (floor(numeric) != numeric || fabs(numeric) > 9007199254740991.0)))
            goto invalid;
        const cJSON *minimum = cJSON_GetObjectItemCaseSensitive(spec, "minimum");
        const cJSON *maximum = cJSON_GetObjectItemCaseSensitive(spec, "maximum");
        if ((cJSON_IsNumber(minimum) && numeric < minimum->valuedouble) ||
            (cJSON_IsNumber(maximum) && numeric > maximum->valuedouble))
            goto invalid;
        value = cJSON_CreateNumber(numeric);
    } else if (field_string(spec, "type", "boolean") &&
               (!strcmp(text, "true") || !strcmp(text, "false"))) {
        value = cJSON_CreateBool(!strcmp(text, "true"));
    }
    if (!value || !cJSON_AddItemToObject(destination, key, value)) {
        cJSON_Delete(value);
        goto invalid;
    }
    return 0;
invalid:
    fprintf(stderr, "Unknown, duplicate, or invalid setting: %s %s\n"
                    "Use FLSHDRMS remote APPLICATION -- --help for supported options.\n", option, text);
    return -1;
}

int main(int argc, char **argv)
{
    const char *host = "10.0.2.2", *local = NULL, *mask = "255.255.255.0";
    const char *gateway = "0.0.0.0", *prompt_file = NULL;
    const char *model_id = NULL, *image_file = NULL, *device = NULL;
    const char *pixel_width = NULL, *pixel_height = NULL, *fps = NULL;
    u32_t duration_ms = 0;
    const char *image_path = NULL;
    const char *mode = "native-window", *output_path = NULL;
    const char *error_path = NULL;
    unsigned char *image = NULL, *loaded_prompt = NULL;
    unsigned image_length = 0;
    unsigned long port = 9390, bpp = 0;
    /* ponytail: up to 16 named endpoints; use a dynamic array if more are needed. */
    struct server_endpoint servers[16];
    unsigned server_count = 0, target_server = 0;
    int legacy_address = 0;
    unsigned long check_frames = 4;
    ip4_addr_t ip, netmask, router;
    int check = 0, added = 0, status = 1, list = 0, list_devices = 0, help = 0, first_setting = argc;
    cJSON *message = NULL, *reply = NULL;
    cJSON *selected = NULL, *parameters = NULL, *inputs = NULL;
    uint32_t frames = 0;
    u32_t started;
    if (argc == 1 || (argc == 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h")))) {
        usage();
        return 0;
    }
    if (strcmp(argv[1], "remote")) {
        fprintf(stderr, "Unsupported command '%s'; this client supports remote.\n", argv[1]);
        goto usage;
    }
    for (int i = 2; i < argc; ++i) {
        char *key = argv[i];
        unsigned long value;
        if (!strcmp(key, "--")) {
            first_setting = i + 1;
            break;
        }
        if (!strcmp(key, "--help") || !strcmp(key, "-h")) {
            usage();
            return 0;
        }
        if (!strcmp(key, "--list-models")) {
            list = 1;
            continue;
        }
        if (!strcmp(key, "--list-devices")) {
            list_devices = 1;
            continue;
        }
        if (!strcmp(key, "--check")) {
            check = 1;
            continue;
        }
        if (key[0] != '-' && !model_id) {
            model_id = key;
            continue;
        }
        char *text = strchr(key, '=');
        if (text) *text++ = 0;
        else {
            if (++i == argc) {
                printf("Missing value for option: %s\n", key);
                goto usage;
            }
            text = argv[i];
        }
        if (!strcmp(key, "--host")) { host = text; legacy_address = 1; }
        else if (!strcmp(key, "--server")) {
            if (server_count == 16 || server_endpoint(text, &servers[server_count])) {
                puts("Expected --server NAME=IPv4:PORT (port 1..65535); at most 16 servers.");
                goto usage;
            }
            for (unsigned s = 0; s < server_count; ++s)
                if (!strcmp(servers[s].name, servers[server_count].name)) {
                    puts("Duplicate --server name.");
                    goto usage;
                }
            ++server_count;
        }
        else if (!strcmp(key, "--device") && !device) device = text;
        else if (!strcmp(key, "--ip")) local = text;
        else if (!strcmp(key, "--netmask")) mask = text;
        else if (!strcmp(key, "--gateway")) gateway = text;
        else if (!strcmp(key, "--mode")) mode = text;
        else if (!strcmp(key, "--output-path")) output_path = text;
        else if (!strcmp(key, "--error-path")) error_path = text;
        else if (!strcmp(key, "--port") && !number(text, 1, 65535, &value)) { port = value; legacy_address = 1; }
        else if (!strcmp(key, "--pixel-width") && !pixel_width) pixel_width = text;
        else if (!strcmp(key, "--pixel-height") && !pixel_height) pixel_height = text;
        else if (!strcmp(key, "--fps") && !fps) fps = text;
        else if (!strcmp(key, "--check-frames") && !number(text, 1, 65535, &value)) {
            check = 1;
            check_frames = value;
        }
        else if (!strcmp(key, "--timeout")) {
            char *end;
            errno = 0;
            double seconds = strtod(text, &end);
            if (!strcmp(text, "unbound")) duration_ms = 0;
            else if (!*text || *end || errno || !isfinite(seconds) || seconds <= 0 ||
                     seconds * 1000 > UINT32_MAX) goto usage;
            else duration_ms = ceil(seconds * 1000);
        }
        else if (!strcmp(key, "--network-timeout") && !number(text, 0, 3600, &value)) timeout_seconds = value;
        else if (!strcmp(key, "--packet-int") && !number(text, 0x60, 0x80, &value)) interrupt = value;
        else if (!strcmp(key, "--bpp") && !number(text, 24, 32, &value) &&
                 (value == 24 || value == 32)) bpp = value;
        else {
            printf("Unknown, misplaced, or duplicate runtime option: %s\n", key);
            goto usage;
        }
    }
    if (server_count && legacy_address) {
        puts("--server cannot be combined with --host or --port.");
        goto usage;
    }
    if (device) {
        const char *slash = strchr(device, '/');
        if (slash) {
            int found = 0;
            for (unsigned s = 0; s < server_count; ++s)
                if (strlen(servers[s].name) == (unsigned)(slash - device) &&
                    !strncmp(servers[s].name, device, slash - device)) {
                    target_server = s;
                    found = 1;
                }
            if (!found || !slash[1]) {
                puts("--device must name a configured server and a device ID.");
                goto usage;
            }
            device = slash + 1;
        } else if (server_count > 1) {
            puts("--device must use NAME/ID when multiple servers are configured.");
            goto usage;
        }
        if (!strcmp(device, "cuda")) device = "cuda:0";
    }
    if (!server_count) {
        servers[0] = (struct server_endpoint){NULL, host, port};
        server_count = 1;
    }
    for (int i = first_setting; i < argc; ++i)
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) help = 1;
    if (((list || list_devices) && model_id) || (list && list_devices) ||
        (first_setting < argc && !model_id)) goto usage;
    int recording = !strcmp(mode, "mp4");
    if (!recording && strcmp(mode, "native-window")) goto usage;
    if (recording && model_id && !help && !output_path) {
        fprintf(stderr, "--output-path is required for --mode mp4.\n");
        return 2;
    }
    if ((error_path || check) && !freopen(error_path ? error_path : "ERROR.TXT", "w", stderr)) {
        puts("Cannot open the requested error log.");
        return 1;
    }
    if (check) {
        if (!timeout_seconds) timeout_seconds = 30;
        packet_trace_enable();
    }
    ip_addr_t validated_host;
    if (!ipaddr_aton(servers[target_server].host, &validated_host) || !ip4addr_aton(mask, &netmask) ||
        !ip4addr_aton(gateway, &router) || (local && !ip4addr_aton(local, &ip))) {
        fprintf(stderr, "Use numeric IPv4 addresses for network settings.\n");
        goto cleanup;
    }
    if (!local)
        ip4_addr_set_zero(&ip);
    lwip_init();
    if (!netif_add(&interface, &ip, &netmask, &router, NULL,
                   init_interface, ethernet_input))
        goto cleanup;
    added = 1;
    netif_set_default(&interface);
    netif_set_link_up(&interface);
    netif_set_up(&interface);
    last_progress = sys_now();
    if (!local) {
        if (dhcp_start(&interface) != ERR_OK) {
            network_error("Cannot start DHCP.");
            goto cleanup;
        }
        while (!dhcp_supplied_address(&interface))
            if (wait_progress()) goto cleanup;
    }
    printf("IPv4 %s\n", ip4addr_ntoa(netif_ip4_addr(&interface)));
    int discover_all = list_devices || ((list || !model_id) && !device);
    unsigned catalogs = discover_all ? server_count : 1;
    for (unsigned s = 0; s < catalogs; ++s) {
        const struct server_endpoint *endpoint = &servers[discover_all ? s : target_server];
        if (endpoint->name) printf("Server %s:\n", endpoint->name);
        int device_selection = connect_service(endpoint->host, endpoint->port);
        if (device_selection < 0) goto cleanup;
        if (list_devices || device) {
            if (!device_selection) {
                network_error("This service does not support device discovery or selection.");
                goto cleanup;
            }
            reply = exchange(request("devices"), "devices");
            if (!reply) goto cleanup;
            const cJSON *devices = cJSON_GetObjectItemCaseSensitive(reply, "devices");
            const cJSON *default_device = cJSON_GetObjectItemCaseSensitive(reply, "default_device");
            if (!cJSON_IsArray(devices) || !cJSON_GetArraySize(devices) ||
                !cJSON_IsString(default_device) || !default_device->valuestring[0]) {
                network_error("Invalid device catalog.");
                goto cleanup;
            }
            int found = device == NULL || list_devices, default_found = 0;
            const cJSON *entry;
            if (list_devices) printf("Server devices (default %s%s%s):\n",
                                     endpoint->name ? endpoint->name : "", endpoint->name ? "/" : "",
                                     default_device->valuestring);
            cJSON_ArrayForEach(entry, devices) {
                const cJSON *id = cJSON_GetObjectItemCaseSensitive(entry, "id");
                const cJSON *name = cJSON_GetObjectItemCaseSensitive(entry, "name");
                const cJSON *kind = cJSON_GetObjectItemCaseSensitive(entry, "type");
                const cJSON *memory = cJSON_GetObjectItemCaseSensitive(entry, "total_memory");
                const cJSON *busy = cJSON_GetObjectItemCaseSensitive(entry, "busy");
                if (!cJSON_IsString(id) || !id->valuestring[0] || !cJSON_IsString(name) ||
                    !cJSON_IsString(kind) || !cJSON_IsBool(busy) ||
                    (!cJSON_IsNull(memory) && (!cJSON_IsNumber(memory) ||
                     !isfinite(memory->valuedouble) || memory->valuedouble < 0 ||
                     floor(memory->valuedouble) != memory->valuedouble))) {
                    network_error("Invalid device catalog entry.");
                    goto cleanup;
                }
                for (const cJSON *previous = devices->child; previous != entry; previous = previous->next) {
                    if (field_string(previous, "id", id->valuestring)) {
                        network_error("Duplicate device catalog ID.");
                        goto cleanup;
                    }
                }
                if (!strcmp(default_device->valuestring, id->valuestring)) default_found = 1;
                if (device && !strcmp(device, id->valuestring)) found = 1;
                if (list_devices) {
                    printf("  %s%s%s: %s (%s)", endpoint->name ? endpoint->name : "",
                           endpoint->name ? "/" : "", id->valuestring, name->valuestring, kind->valuestring);
                    if (cJSON_IsNumber(memory)) printf("; %.0f MiB", memory->valuedouble / (1024 * 1024));
                    printf("; %s\n", cJSON_IsTrue(busy) ? "busy" : "available");
                }
            }
            if (!default_found || !found) {
                network_error(default_found ? "Service does not offer the requested device." : "Invalid default device.");
                goto cleanup;
            }
            cJSON_Delete(reply);
            reply = NULL;
            if (list_devices) {
                if (s + 1 < catalogs) { disconnect_stream(); continue; }
                status = 0;
                goto cleanup;
            }
        }
        reply = exchange(request("models"), "models");
        if (!reply) goto cleanup;
        const cJSON *models = cJSON_GetObjectItemCaseSensitive(reply, "models");
        if (!cJSON_IsArray(models)) {
            network_error("Invalid model catalog.");
            goto cleanup;
        }
        printf("Models:");
        const cJSON *model;
        cJSON_ArrayForEach(model, models) {
            const cJSON *id = cJSON_GetObjectItemCaseSensitive(model, "id");
            if (!cJSON_IsString(id)) {
                network_error("Invalid model catalog entry.");
                goto cleanup;
            }
            printf(" %s", id->valuestring);
            if (model_id && !strcmp(id->valuestring, model_id)) {
                if (selected) {
                    network_error("Duplicate model catalog ID.");
                    goto cleanup;
                }
                selected = cJSON_Duplicate(model, 1);
            }
        }
        printf("\n");
        if (list || !model_id) {
            cJSON_Delete(reply);
            reply = NULL;
            if (s + 1 < catalogs) { disconnect_stream(); continue; }
            puts("Select a model with FLSHDRMS remote APPLICATION -- [application options].");
            status = 0;
            goto cleanup;
        }
        if (!selected) {
            fprintf(stderr, "Service does not offer model '%s'.\n", model_id);
            goto cleanup;
        }
        if (!cJSON_IsObject(cJSON_GetObjectItemCaseSensitive(selected, "parameters")) ||
            !cJSON_IsObject(cJSON_GetObjectItemCaseSensitive(selected, "inputs"))) {
            network_error("Invalid model setting schemas.");
            goto cleanup;
        }
        cJSON_Delete(reply);
        reply = NULL;
        break;
    }
    if (help) {
        printf("Model: %s\n", model_id);
        describe_settings(cJSON_GetObjectItemCaseSensitive(selected, "parameters"), "Runtime options before --", 1);
        describe_settings(cJSON_GetObjectItemCaseSensitive(selected, "parameters"), "Application options after --", 0);
        describe_settings(cJSON_GetObjectItemCaseSensitive(selected, "inputs"), "Inputs after --", 0);
        const cJSON *requirement = cJSON_GetObjectItemCaseSensitive(selected, "input_image");
        if (cJSON_IsString(requirement)) printf("Image input: %s (--image-path FILE or --initial-rgb FILE)\n", requirement->valuestring);
        status = 0;
        goto cleanup;
    }
    parameters = cJSON_CreateObject();
    inputs = cJSON_CreateObject();
    if (!parameters || !inputs) goto cleanup;
    if ((pixel_width && setting(parameters, inputs, selected, "--width", pixel_width)) ||
        (pixel_height && setting(parameters, inputs, selected, "--height", pixel_height)) ||
        (fps && setting(parameters, inputs, selected, "--fps", fps))) goto cleanup;
    for (int i = first_setting; i < argc; ++i) {
        char *key = argv[i], *text = strchr(key, '=');
        if (text) *text++ = 0;
        if (!strcmp(key, "--width") || !strcmp(key, "--height") || !strcmp(key, "--fps")) {
            fprintf(stderr, "Use --pixel-width, --pixel-height, or --fps before --.\n");
            goto cleanup;
        }
        if (!strcmp(key, "--compile") || !strcmp(key, "--no-compile")) {
            if (text) {
                fprintf(stderr, "%s does not take a value.\n", key);
                goto cleanup;
            }
            if (setting(parameters, inputs, selected, "--compile",
                        !strcmp(key, "--compile") ? "true" : "false")) goto cleanup;
            continue;
        }
        if (!text) {
            if (++i == argc) {
                fprintf(stderr, "Missing value for setting: %s\n", key);
                goto cleanup;
            }
            text = argv[i];
        }
        if (!strcmp(key, "--image-path") && !image_path) image_path = text;
        else if (!strcmp(key, "--initial-rgb") && !image_file) image_file = text;
        else if (!strcmp(key, "--prompt-file") && !prompt_file) prompt_file = text;
        else if (setting(parameters, inputs, selected, key, text)) goto cleanup;
    }
    if (prompt_file) {
        unsigned length;
        loaded_prompt = load_file(prompt_file, 8192, &length);
        if (!loaded_prompt) goto cleanup;
        if (memchr(loaded_prompt, 0, length)) {
            network_error("Prompt file contains NUL.");
            goto cleanup;
        }
        if (setting(parameters, inputs, selected, "--prompt", (char *)loaded_prompt)) goto cleanup;
    }
    if (image_path && image_file) {
        fprintf(stderr, "Use --image-path or --initial-rgb, not both.\n");
        goto cleanup;
    }
    int has_image = image_file != NULL || image_path != NULL;
    if ((!field_string(selected, "input_image", "none") &&
         !field_string(selected, "input_image", "optional_rgb24") &&
         !field_string(selected, "input_image", "required_rgb24")) ||
        (field_string(selected, "input_image", "required_rgb24") && !has_image) ||
        (field_string(selected, "input_image", "none") && has_image)) {
        network_error("The selected model's image requirement does not match --image-path/--initial-rgb.");
        goto cleanup;
    }
    if (has_image) {
        unsigned width = image_dimension(parameters, selected, "width");
        unsigned height = image_dimension(parameters, selected, "height");
        if (!width || !height || width * height * 3 > MAX_PAYLOAD) {
            network_error("Invalid image dimensions in the model settings.");
            goto cleanup;
        }
        if (image_path) {
            image = image_load(image_path, width, height);
            image_length = width * height * 3;
        } else {
            image = load_file(image_file, MAX_PAYLOAD, &image_length);
        }
        if (!image) goto cleanup;
        if (image_length != width * height * 3) {
            network_error("Raw RGB image must match the requested width and height.");
            goto cleanup;
        }
    }
    message = request("open");
    cJSON_AddStringToObject(message, "model", model_id);
    if (device) cJSON_AddStringToObject(message, "device", device);
    cJSON_AddItemToObject(message, "parameters", cJSON_Duplicate(parameters, 1));
    run_started = sys_now();
    run_limit_ms = duration_ms;
    reply = exchange_payload(message, "session", image, image_length);
    free(image);
    image = NULL;
    if (!reply) goto cleanup;
    if (!field_string(reply, "model", model_id) || (device && !field_string(reply, "device", device))) {
        network_error("Service opened the wrong model or device.");
        goto cleanup;
    }
    char *description = cJSON_PrintUnformatted(reply);
    if (!description) goto cleanup;
    printf("Session: %s\n", description);
    free(description);
    cJSON_Delete(reply);
    reply = NULL;
    printf("Output mode: %s\n", mode);
    started = sys_now();
    while (!stopped) {
        message = request("step");
        cJSON_AddNumberToObject(message, "count", 1);
        cJSON_AddItemToObject(message, "inputs", cJSON_Duplicate(inputs, 1));
        reply = exchange(message, "frames");
        if (!reply) goto cleanup;
        if (field_string(reply, "type", "finished")) {
            if (!field_number(reply, "next_frame", frames)) {
                network_error("Invalid end-of-stream frame index.");
                goto cleanup;
            }
            if (check) {
                network_error("Model finished before the requested frame check completed.");
                goto cleanup;
            }
            cJSON_Delete(reply);
            reply = NULL;
            if (recording) {
                puts("Generation finished.");
                break;
            }
            puts("Generation finished. Press Escape to exit.");
            while (!stopped)
                if (poll_network()) goto cleanup;
            break;
        }
        const cJSON *width = cJSON_GetObjectItemCaseSensitive(reply, "width");
        const cJSON *height = cJSON_GetObjectItemCaseSensitive(reply, "height");
        const cJSON *fps = cJSON_GetObjectItemCaseSensitive(reply, "fps");
        if (!cJSON_IsNumber(width) || !cJSON_IsNumber(height) ||
            width->valuedouble < 1 || width->valuedouble > 65535 ||
            height->valuedouble < 1 || height->valuedouble > 65535 ||
            floor(width->valuedouble) != width->valuedouble ||
            floor(height->valuedouble) != height->valuedouble) {
            network_error("Invalid frame dimensions.");
            goto cleanup;
        }
        unsigned frame_width = width->valuedouble, frame_height = height->valuedouble;
        if (!field_string(reply, "format", "rgb24") ||
            !field_number(reply, "first_frame", frames) || !field_number(reply, "frame_count", 1) ||
            !cJSON_IsNumber(fps) || fps->valuedouble < 1 || fps->valuedouble > 120 ||
            floor(fps->valuedouble) != fps->valuedouble ||
            (uint64_t)frame_width * frame_height * 3 != payload_size) {
            network_error("Invalid RGB frame metadata.");
            goto cleanup;
        }
        unsigned frame_fps = fps->valuedouble;
        cJSON_Delete(reply);
        reply = NULL;
        if (recording) {
            if (!frames || (u32_t)(sys_now() - last_output) >= 1000) {
                printf("Encoding frame %lu on DOS...\n", (unsigned long)frames + 1);
                fflush(stdout);
                last_output = sys_now();
            }
            if (mp4_write_frame(output_path, rgb, frame_width, frame_height, frame_fps)) {
                failed = 1;
                goto cleanup;
            }
        } else {
            if (!graphics_active) {
                puts("Displaying frames locally. Press Escape to exit.");
                if (vbe_init(bpp)) goto cleanup;
                graphics_active = 1;
            }
            vbe_draw(rgb, frame_width, frame_height);
        }
        ++frames;
        if (check)
            printf("Frame %lu received and %s; packets=%lu rejected=%u\n",
                   (unsigned long)frames, recording ? "encoded" : "displayed",
                   packet_received(), packet_dropped());
        if (check && frames == check_frames) {
            if (!recording && vbe_save()) {
                network_error("Cannot save framebuffer check.");
                goto cleanup;
            }
            break;
        }
        if (poll_network()) goto cleanup;
    }
    reply = exchange(request("close"), "closed");
    if (!reply) goto cleanup;
    printf("PASS: model-service session; frames=%lu elapsed_ms=%lu packets=%lu rejected=%u\n",
           (unsigned long)frames, (unsigned long)(sys_now() - started),
           packet_received(), packet_dropped());
    status = 0;
cleanup:
    if (mp4_close()) {
        failed = 1;
        status = 1;
    } else if (recording && frames && !failed) {
        printf("MP4 saved: %s\n", output_path);
    }
    cJSON_Delete(reply);
    cJSON_Delete(selected);
    cJSON_Delete(parameters);
    cJSON_Delete(inputs);
    vbe_shutdown();
    disconnect_stream();
    if (added) {
        if (!local) dhcp_release_and_stop(&interface);
        netif_set_down(&interface);
        netif_remove(&interface);
    }
    packet_close();
    free(rgb);
    free(image);
    free(loaded_prompt);
    if (stopped && !failed) {
        puts(run_expired ? "Stopped at --timeout." : "Stopped with Escape.");
        return 0;
    }
    return status;
usage:
    usage();
    return 2;
}
