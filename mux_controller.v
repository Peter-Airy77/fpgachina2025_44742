// ================================================================
// Module: mux_controller
// Description: 16x16 Matrix multiplexer controller
// Scans all 256 points in the pressure sensor matrix
// ================================================================
module mux_controller(
    input               clk,            // System clock (50MHz)
    input               rst_n,          // Active low reset
    // Multiplexer control signals
    output reg [3:0]    row_sel,        // Row select (4-bit, selects 0-15)
    output reg [3:0]    col_sel,        // Column select (4-bit, selects 0-15)
    output reg          mux_valid,      // Multiplexer output valid
    // Scan control
    output reg [8:0]    scan_count,     // Scan counter (0-255, needs 9 bits)
    output reg          frame_start,    // Frame start signal
    output reg          frame_done,     // Frame done signal
    // ADC protocol signals
    input               adc_ready,      // ADC ready signal
    output reg          adc_start       // ADC conversion start
);

// Parameters
parameter SCAN_DELAY = 8'd10;          // Multiplexer stabilization delay
parameter ADC_SETUP  = 8'd5;           // ADC setup time

// State machine parameters
parameter [2:0] IDLE        = 3'b000;    // Idle state
parameter [2:0] SET_ROW_COL = 3'b001;    // Set row and column select
parameter [2:0] WAIT_STABLE = 3'b010;    // Wait for multiplexer to stabilize
parameter [2:0] START_ADC   = 3'b011;    // Start ADC conversion
parameter [2:0] WAIT_ADC    = 3'b100;    // Wait for ADC completion
parameter [2:0] NEXT_POINT  = 3'b101;    // Prepare for next point

// Register declarations
reg [2:0] current_state;
reg [2:0] next_state;
reg [7:0] delay_counter;                 // Delay counter
reg [3:0] current_row;                
reg [3:0] current_col;

//*****************************************************
//**                    main code
//*****************************************************

// State register
always @(posedge clk or negedge rst_n) begin
    if(!rst_n)
        current_state <= IDLE;
    else
        current_state <= next_state;
end

// State transition logic
always @(*) begin
    case(current_state)
        IDLE: begin
            next_state = SET_ROW_COL;
        end
        
        SET_ROW_COL: begin
            next_state = WAIT_STABLE;
        end
        
        WAIT_STABLE: begin
            if(delay_counter >= SCAN_DELAY)
                next_state = START_ADC;
            else
                next_state = WAIT_STABLE;
        end
        
        START_ADC: begin
            if(adc_ready)
                next_state = WAIT_ADC;
            else
                next_state = START_ADC;
        end
        
        WAIT_ADC: begin
            if(delay_counter >= ADC_SETUP)
                next_state = NEXT_POINT;
            else
                next_state = WAIT_ADC;
        end
        
        NEXT_POINT: begin
            if(scan_count == 9'd255)
                next_state = IDLE;
            else
                next_state = SET_ROW_COL;
        end
        
        default: next_state = IDLE;
    endcase
end

// Delay counter
always @(posedge clk or negedge rst_n) begin
    if(!rst_n)
        delay_counter <= 8'b0;
    else if(current_state == WAIT_STABLE || current_state == WAIT_ADC)
        delay_counter <= delay_counter + 1'b1;
    else
        delay_counter <= 8'b0;
end

// Row scan counter
always @(posedge clk or negedge rst_n) begin
    if(!rst_n)
        current_row <= 4'b0;
    else if(current_state == NEXT_POINT) begin
        if(current_col == 4'd15) begin
            if(current_row == 4'd15)
                current_row <= 4'b0;
            else
                current_row <= current_row + 1'b1;
        end
    end
end

// Column scan counter
always @(posedge clk or negedge rst_n) begin
    if(!rst_n)
        current_col <= 4'b0;
    else if(current_state == NEXT_POINT) begin
        if(current_col == 4'd15)
            current_col <= 4'b0;
        else
            current_col <= current_col + 1'b1;
    end
end

// Total scan counter (0-255)
always @(posedge clk or negedge rst_n) begin
    if(!rst_n)
        scan_count <= 9'b0;
    else if(current_state == NEXT_POINT) begin
        if(scan_count == 9'd255)
            scan_count <= 9'b0;
        else
            scan_count <= scan_count + 1'b1;
    end
end

// Output logic
always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        row_sel <= 4'b0;
        col_sel <= 4'b0;
        mux_valid <= 1'b0;
        frame_start <= 1'b0;
        frame_done <= 1'b0;
        adc_start <= 1'b0;
    end
    else begin
        case(current_state)
            IDLE: begin
                row_sel <= 4'b0;
                col_sel <= 4'b0;
                mux_valid <= 1'b0;
                frame_start <= 1'b1;  // Start new frame
                frame_done <= 1'b0;
                adc_start <= 1'b0;
            end
            
            SET_ROW_COL: begin
                row_sel <= current_row;
                col_sel <= current_col;
                mux_valid <= 1'b1;
                frame_start <= 1'b0;
                frame_done <= 1'b0;
                adc_start <= 1'b0;
            end
            
            WAIT_STABLE: begin
                mux_valid <= 1'b1;
                adc_start <= 1'b0;
            end
            
            START_ADC: begin
                adc_start <= 1'b1;
            end
            
            WAIT_ADC: begin
                adc_start <= 1'b0;
            end
            
            NEXT_POINT: begin
                if(scan_count == 9'd255)
                    frame_done <= 1'b1;  // Frame complete
                else
                    frame_done <= 1'b0;
            end
            
            default: begin
                mux_valid <= 1'b0;
                adc_start <= 1'b0;
                frame_start <= 1'b0;
                frame_done <= 1'b0;
            end
        endcase
    end
end

endmodule

