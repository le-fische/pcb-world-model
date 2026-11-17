// ============================================================================
// Module: quad_ws2812_spi
// Target: Lattice iCE40UP5K-SG48 / OpenLane SKY130 (Verilog-2001 Synthesizable)
// Description:
//   - 8-channel 4x hardware quadrature encoder decoders (16-bit counters)
//   - 8-channel parallel 800kHz WS2812B NRZ LED drivers (24-bit GRB per channel)
//   - Synchronized 32-bit SPI Mode-0 slave bridge for ESP32-C3 host MCU
// ============================================================================
`timescale 1ns / 1ps

module quad_ws2812_spi #(
    parameter [15:0] BIT_CYCLES   = 16'd60,
    parameter [15:0] T0H_CYCLES   = 16'd19,
    parameter [15:0] T1H_CYCLES   = 16'd38,
    parameter [15:0] RESET_CYCLES = 16'd2640
) (
    input  wire       clk,
    input  wire       rst_n,
    // SPI Mode-0 Slave Interface (from ESP32-C3)
    input  wire       spi_sck,
    input  wire       spi_mosi,
    output reg        spi_miso,
    input  wire       spi_cs_n,
    output reg        irq_n,
    // 8-Channel Quadrature Encoder Inputs
    input  wire [7:0] enc_a,
    input  wire [7:0] enc_b,
    // 8-Channel WS2812B NRZ Outputs
    output reg  [7:0] ws_out
);

    // ------------------------------------------------------------------------
    // 1. 2-FF CDC Input Synchronizers (SPI + Encoders)
    // ------------------------------------------------------------------------
    reg [1:0] sck_sync;
    reg [1:0] cs_sync;
    reg [1:0] mosi_sync;
    reg [7:0] enc_a_d1, enc_a_d2, enc_a_prev;
    reg [7:0] enc_b_d1, enc_b_d2, enc_b_prev;

    wire sck_rise = (sck_sync == 2'b01);
    wire sck_fall = (sck_sync == 2'b10);
    wire cs_active = ~cs_sync[1];
    wire cs_fall = (cs_sync == 2'b10);

    // ------------------------------------------------------------------------
    // 2. 8-Channel 4x Hardware Quadrature Encoder Decoders
    // ------------------------------------------------------------------------
    reg [15:0] enc_cnt [0:7];
    integer ch_i;
    reg clear_irq_pulse;
    wire [7:0] enc_changed = (enc_a_d2 ^ enc_a_prev) | (enc_b_d2 ^ enc_b_prev);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            enc_a_d1   <= 8'h00;
            enc_a_d2   <= 8'h00;
            enc_a_prev <= 8'h00;
            enc_b_d1   <= 8'h00;
            enc_b_d2   <= 8'h00;
            enc_b_prev <= 8'h00;
            irq_n      <= 1'b1;
            for (ch_i = 0; ch_i < 8; ch_i = ch_i + 1) begin
                enc_cnt[ch_i] <= 16'h0000;
            end
        end else begin
            enc_a_d1   <= enc_a;
            enc_a_d2   <= enc_a_d1;
            enc_a_prev <= enc_a_d2;
            enc_b_d1   <= enc_b;
            enc_b_d2   <= enc_b_d1;
            enc_b_prev <= enc_b_d2;

            for (ch_i = 0; ch_i < 8; ch_i = ch_i + 1) begin
                case ({enc_a_prev[ch_i], enc_b_prev[ch_i], enc_a_d2[ch_i], enc_b_d2[ch_i]})
                    4'b0001, 4'b0111, 4'b1110, 4'b1000: begin
                        enc_cnt[ch_i] <= enc_cnt[ch_i] + 16'd1;
                    end
                    4'b0010, 4'b1011, 4'b1101, 4'b0100: begin
                        enc_cnt[ch_i] <= enc_cnt[ch_i] - 16'd1;
                    end
                    default: begin
                        enc_cnt[ch_i] <= enc_cnt[ch_i];
                    end
                endcase
            end

            if (clear_irq_pulse) begin
                irq_n <= 1'b1;
            end else if (|enc_changed) begin
                irq_n <= 1'b0;
            end
        end
    end

    // ------------------------------------------------------------------------
    // 3. SPI Mode-0 32-bit Register Slave ([31]=W, [30:24]=Addr, [23:0]=Data)
    // ------------------------------------------------------------------------
    reg [22:0] rx_shift;
    reg [23:0] tx_shift;
    reg [5:0]  spi_bit_cnt;
    reg [7:0]  spi_cmd;
    reg [23:0] ws_color [0:7];
    reg        ws_start_pulse;
    reg        ws_busy;

    wire [7:0] next_cmd = {rx_shift[6:0], mosi_sync[1]};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sck_sync        <= 2'b00;
            cs_sync         <= 2'b11;
            mosi_sync       <= 2'b00;
            rx_shift        <= 23'h0;
            tx_shift        <= 24'h0;
            spi_bit_cnt     <= 6'd0;
            spi_cmd         <= 8'h00;
            spi_miso        <= 1'b0;
            ws_start_pulse  <= 1'b0;
            clear_irq_pulse <= 1'b0;
            for (ch_i = 0; ch_i < 8; ch_i = ch_i + 1) begin
                ws_color[ch_i] <= 24'h000000;
            end
        end else begin
            sck_sync        <= {sck_sync[0], spi_sck};
            cs_sync         <= {cs_sync[0], spi_cs_n};
            mosi_sync       <= {mosi_sync[0], spi_mosi};
            ws_start_pulse  <= 1'b0;
            clear_irq_pulse <= 1'b0;

            if (cs_fall || !cs_active) begin
                spi_bit_cnt <= 6'd0;
                rx_shift    <= 23'h0;
                spi_miso    <= 1'b0;
            end else begin
                if (sck_rise) begin
                    rx_shift    <= {rx_shift[21:0], mosi_sync[1]};
                    spi_bit_cnt <= spi_bit_cnt + 6'd1;

                    if (spi_bit_cnt == 6'd7) begin
                        spi_cmd <= next_cmd;
                        if (next_cmd[6:0] < 7'h08) begin
                            tx_shift <= {8'h00, enc_cnt[next_cmd[2:0]]};
                        end else if (next_cmd[6:0] >= 7'h10 && next_cmd[6:0] <= 7'h17) begin
                            tx_shift <= ws_color[next_cmd[2:0]];
                        end else begin
                            tx_shift <= {16'hA500, 6'b0, ~irq_n, ws_busy};
                        end
                    end

                    if (spi_bit_cnt == 6'd31) begin
                        if (spi_cmd[7]) begin
                            if (spi_cmd[6:0] >= 7'h10 && spi_cmd[6:0] <= 7'h17) begin
                                ws_color[spi_cmd[2:0]] <= {rx_shift[22:0], mosi_sync[1]};
                            end else if (spi_cmd[6:0] == 7'h20) begin
                                if (mosi_sync[1]) begin
                                    ws_start_pulse <= 1'b1;
                                end
                                if (rx_shift[0]) begin
                                    clear_irq_pulse <= 1'b1;
                                end
                            end
                        end
                    end
                end else if (sck_fall && spi_bit_cnt >= 6'd8) begin
                    spi_miso <= tx_shift[23];
                    tx_shift <= {tx_shift[22:0], 1'b0};
                end
            end
        end
    end

    // ------------------------------------------------------------------------
    // 4. 8-Channel Parallel WS2812B NRZ Timing Engine (800 kHz)
    // ------------------------------------------------------------------------
    localparam [1:0] WS_IDLE  = 2'd0;
    localparam [1:0] WS_SEND  = 2'd1;
    localparam [1:0] WS_RESET = 2'd2;

    reg [1:0]  ws_state;
    reg [15:0] ws_cyc_cnt;
    reg [4:0]  ws_bit_idx;
    reg [23:0] ws_latched [0:7];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ws_state   <= WS_IDLE;
            ws_cyc_cnt <= 16'd0;
            ws_bit_idx <= 5'd0;
            ws_out     <= 8'h00;
            ws_busy    <= 1'b0;
            for (ch_i = 0; ch_i < 8; ch_i = ch_i + 1) begin
                ws_latched[ch_i] <= 24'h000000;
            end
        end else begin
            case (ws_state)
                WS_IDLE: begin
                    ws_out  <= 8'h00;
                    ws_busy <= 1'b0;
                    if (ws_start_pulse) begin
                        for (ch_i = 0; ch_i < 8; ch_i = ch_i + 1) begin
                            ws_latched[ch_i] <= ws_color[ch_i];
                        end
                        ws_state   <= WS_SEND;
                        ws_cyc_cnt <= 16'd0;
                        ws_bit_idx <= 5'd23;
                        ws_busy    <= 1'b1;
                    end
                end

                WS_SEND: begin
                    for (ch_i = 0; ch_i < 8; ch_i = ch_i + 1) begin
                        if (ws_latched[ch_i][ws_bit_idx]) begin
                            ws_out[ch_i] <= (ws_cyc_cnt < T1H_CYCLES);
                        end else begin
                            ws_out[ch_i] <= (ws_cyc_cnt < T0H_CYCLES);
                        end
                    end

                    if (ws_cyc_cnt + 16'd1 >= BIT_CYCLES) begin
                        ws_cyc_cnt <= 16'd0;
                        if (ws_bit_idx == 5'd0) begin
                            ws_state <= WS_RESET;
                            ws_out   <= 8'h00;
                        end else begin
                            ws_bit_idx <= ws_bit_idx - 5'd1;
                        end
                    end else begin
                        ws_cyc_cnt <= ws_cyc_cnt + 16'd1;
                    end
                end

                WS_RESET: begin
                    ws_out <= 8'h00;
                    if (ws_cyc_cnt + 16'd1 >= RESET_CYCLES) begin
                        ws_cyc_cnt <= 16'd0;
                        ws_state   <= WS_IDLE;
                        ws_busy    <= 1'b0;
                    end else begin
                        ws_cyc_cnt <= ws_cyc_cnt + 16'd1;
                    end
                end

                default: begin
                    ws_state <= WS_IDLE;
                end
            endcase
        end
    end

endmodule
