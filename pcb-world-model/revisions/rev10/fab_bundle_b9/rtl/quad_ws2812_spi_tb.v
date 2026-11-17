// ============================================================================
// Self-Checking Testbench for quad_ws2812_spi
// Verifies: (1) 4x quadrature forward/reverse decoding, (2) SPI Mode-0 R/W,
//           (3) WS2812B T0H vs T1H cycle-exact NRZ pulse timing, (4) IRQ clear
// ============================================================================
`timescale 1ns / 1ps

module quad_ws2812_spi_tb;
    reg        clk = 1'b0;
    reg        rst_n = 1'b0;
    reg        spi_sck = 1'b0;
    reg        spi_mosi = 1'b0;
    wire       spi_miso;
    reg        spi_cs_n = 1'b1;
    wire       irq_n;
    reg  [7:0] enc_a = 8'h00;
    reg  [7:0] enc_b = 8'h00;
    wire [7:0] ws_out;

    quad_ws2812_spi #(
        .BIT_CYCLES(60),
        .T0H_CYCLES(19),
        .T1H_CYCLES(38),
        .RESET_CYCLES(120)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .spi_sck(spi_sck),
        .spi_mosi(spi_mosi),
        .spi_miso(spi_miso),
        .spi_cs_n(spi_cs_n),
        .irq_n(irq_n),
        .enc_a(enc_a),
        .enc_b(enc_b),
        .ws_out(ws_out)
    );

    always #10 clk = ~clk;

    integer errors = 0;
    integer checks = 0;
    reg [23:0] spi_rx_val;
    integer hi_ch0;
    integer hi_ch1;
    integer cyc;

    task step_enc_fwd;
        input integer ch;
        begin
            enc_b[ch] = 1'b1; repeat (8) @(posedge clk);
            enc_a[ch] = 1'b1; repeat (8) @(posedge clk);
            enc_b[ch] = 1'b0; repeat (8) @(posedge clk);
            enc_a[ch] = 1'b0; repeat (8) @(posedge clk);
        end
    endtask

    task step_enc_rev;
        input integer ch;
        begin
            enc_a[ch] = 1'b1; repeat (8) @(posedge clk);
            enc_b[ch] = 1'b1; repeat (8) @(posedge clk);
            enc_a[ch] = 1'b0; repeat (8) @(posedge clk);
            enc_b[ch] = 1'b0; repeat (8) @(posedge clk);
        end
    endtask

    task spi_xfer32;
        input  [7:0]  cmd;
        input  [23:0] data_in;
        output [23:0] data_out;
        reg    [31:0] frame;
        integer i;
        begin
            frame = {cmd, data_in};
            data_out = 24'h0;
            spi_sck = 1'b0;
            spi_cs_n = 1'b0;
            repeat (4) @(posedge clk);
            for (i = 31; i >= 0; i = i - 1) begin
                spi_mosi = frame[i];
                repeat (4) @(posedge clk);
                spi_sck = 1'b1;
                repeat (4) @(posedge clk);
                if (i <= 23) begin
                    data_out[i] = spi_miso;
                end
                spi_sck = 1'b0;
                repeat (4) @(posedge clk);
            end
            spi_cs_n = 1'b1;
            repeat (6) @(posedge clk);
        end
    endtask

    initial begin
        rst_n = 1'b0;
        repeat (6) @(posedge clk);
        rst_n = 1'b1;
        repeat (6) @(posedge clk);

        // Check 1: Quadrature forward (+4 on ch0) and reverse (-4 on ch3)
        step_enc_fwd(0);
        step_enc_rev(3);

        checks = checks + 1;
        if (irq_n !== 1'b0) begin
            $display("FAIL: irq_n did not assert after encoder step");
            errors = errors + 1;
        end

        spi_xfer32(8'h00, 24'h000000, spi_rx_val);
        checks = checks + 1;
        if (spi_rx_val[15:0] !== 16'd4) begin
            $display("FAIL: enc[0] expected 4, got %0d", spi_rx_val[15:0]);
            errors = errors + 1;
        end

        spi_xfer32(8'h03, 24'h000000, spi_rx_val);
        checks = checks + 1;
        if (spi_rx_val[15:0] !== 16'hFFFC) begin
            $display("FAIL: enc[3] expected -4 (0xFFFC), got 0x%04x", spi_rx_val[15:0]);
            errors = errors + 1;
        end

        // Check 2: Write WS2812 channel 0 = 0xFF0000 (MSB=1 -> T1H=38), channel 1 = 0x000000 (MSB=0 -> T0H=19)
        spi_xfer32(8'h90, 24'hFF0000, spi_rx_val);
        spi_xfer32(8'h91, 24'h000000, spi_rx_val);

        // Trigger WS2812 transmission (bit0=start) + clear IRQ (bit1=clear)
        spi_xfer32(8'hA0, 24'h000003, spi_rx_val);

        checks = checks + 1;
        if (irq_n !== 1'b1) begin
            $display("FAIL: irq_n did not clear after 0xA0 write");
            errors = errors + 1;
        end

        // Measure first bit pulse width on ws_out[0] and ws_out[1]
        hi_ch0 = 0;
        hi_ch1 = 0;
        @(posedge ws_out[0]);
        for (cyc = 0; cyc < 60; cyc = cyc + 1) begin
            @(posedge clk);
            if (ws_out[0]) hi_ch0 = hi_ch0 + 1;
            if (ws_out[1]) hi_ch1 = hi_ch1 + 1;
        end

        checks = checks + 1;
        if (hi_ch0 !== 38 || hi_ch1 !== 19) begin
            $display("FAIL: WS2812 timing mismatch hi_ch0=%0d (exp 38), hi_ch1=%0d (exp 19)", hi_ch0, hi_ch1);
            errors = errors + 1;
        end

        if (errors == 0) begin
            $display("SIM_PASS: module=quad_ws2812_spi checks=%0d errors=0", checks);
            $finish;
        end else begin
            $display("SIM_FAIL: module=quad_ws2812_spi checks=%0d errors=%0d", checks, errors);
            $fatal(1);
        end
    end
endmodule
