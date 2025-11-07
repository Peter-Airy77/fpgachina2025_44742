// ================================================================
// Module: hs_dual_ad
// Description: High-speed dual ADC controller
// ================================================================
module hs_dual_ad(
    input          hs_clk,           // System clock (50MHz)
    input          sys_rst_n,        // System reset
    // Multiplexer control interface
    input          mux_valid,        // Multiplexer output valid
    input          adc_start,        // ADC conversion start
    // ADC hardware interface
    input   [7:0]  ad_data_1,        // ADC data input
    input          ad_otr_1,         // ADC over-range flag
    output         ad_clk_1,         // ADC clock
    output         ad_oe_1,          // ADC output enable
    // Output signals
    output  [7:0]  ad_data_out,      // ADC data output
    output         ad_data_valid,    // Data valid signal
    output         adc_ready,        // ADC ready signal
    output         ad_error          // ADC error indicator
);

// Register declarations
reg [7:0] ad_data_sync;             // ADC data sync register
reg       ad_otr_sync;              // Over-range flag sync
reg       data_valid_reg;           // Data valid flag
reg       adc_ready_reg;            // ADC ready status
reg [2:0] adc_state;                // ADC state machine
reg [1:0] wait_counter;              

// State machine parameters
parameter ADC_IDLE   = 3'b000;      // Idle state
parameter ADC_WAIT   = 3'b001;      // Wait for ADC to stabilize
parameter ADC_VALID  = 3'b010;      // Data valid state

// Parameters
parameter WAIT_CYCLES = 2'd2;        // Wait for 2 clock cycles

//*****************************************************
//**                    main code
//*****************************************************

// ADC control signals
assign ad_oe_1 = 1'b0;               // Output enable (active low)
assign ad_clk_1 = hs_clk;            // ADC clock (50MHz)

// Output assignments
assign ad_data_out = ad_data_sync;
assign ad_data_valid = data_valid_reg;
assign adc_ready = adc_ready_reg;
assign ad_error = ad_otr_sync;

// ADC state machine
always @(posedge hs_clk or negedge sys_rst_n) begin
    if(!sys_rst_n) begin
        adc_state <= ADC_IDLE;
        adc_ready_reg <= 1'b1;
        data_valid_reg <= 1'b0;
        wait_counter <= 2'b0;
    end
    else begin
        case(adc_state)
            ADC_IDLE: begin
                data_valid_reg <= 1'b0;
                adc_ready_reg <= 1'b1;
                wait_counter <= 2'b0;
                if(adc_start && mux_valid) begin
                    adc_state <= ADC_WAIT;
                    adc_ready_reg <= 1'b0;
                end
            end
            
            ADC_WAIT: begin
                // Wait for specified cycles
                wait_counter <= wait_counter + 2'b1;
                
                // Transition to valid state after wait period
                if(wait_counter >= (WAIT_CYCLES - 1)) begin
                    adc_state <= ADC_VALID;
                    wait_counter <= 2'b0;
                end
                else begin
                    adc_state <= ADC_WAIT;
                end
            end
            
            ADC_VALID: begin
                data_valid_reg <= 1'b1;     // Data is valid
                adc_ready_reg <= 1'b1;      // ADC ready for next conversion
                adc_state <= ADC_IDLE;      // Return to idle
            end
            
            default: begin
                adc_state <= ADC_IDLE;
                adc_ready_reg <= 1'b1;
                data_valid_reg <= 1'b0;
                wait_counter <= 2'b0;
            end
        endcase
    end
end

// ADC data synchronization
// Capture data when entering WAIT state to ensure stable multiplexer output
always @(posedge hs_clk or negedge sys_rst_n) begin
    if(!sys_rst_n) begin
        ad_data_sync <= 8'b0;
        ad_otr_sync <= 1'b0;
    end
    else if(adc_state == ADC_WAIT && wait_counter == 2'b0) begin
        ad_data_sync <= ad_data_1;   // Capture ADC data at start of WAIT state
        ad_otr_sync <= ad_otr_1;     // Sync over-range flag
    end
end

endmodule

