#!/usr/bin/env python3

#####################################################################
# This script presents labels buffer that shows only visible game objects
# (enemies, pickups, exploding barrels etc.), each with unique label.
# OpenCV is used here to display images, install it or remove any
# references to cv2
# Configuration is loaded from "../../scenarios/basic.cfg" file.
# <episodes> number of episodes are played.
# Random combination of buttons is chosen for every action.
# Game variables from state and last reward are printed.
#
# To see the scenario description go to "../../scenarios/README.md"
#####################################################################

import vizdoom as vzd

if __name__ =="__main__":
    game = vzd.DoomGame()
    game.set_doom_game_path("../build/bin/doom.wad")
    game.set_doom_scenario_path("../build/bin/doom.wad")
    game.set_doom_map("E1M1")
    game.set_mode(vzd.Mode.SPECTATOR)
    game.set_episode_start_time(10)
    game.set_episode_timeout(999999999)
    #game.add_game_args("+freelook 1")

    # Use other config file if you wish.
    #game.load_config(args.config)
    game.set_render_hud(True)
    game.set_render_minimal_hud(True)
    game.set_render_decals(True)
    game.set_render_particles(True)
    game.set_render_effects_sprites(True)
    game.set_render_corpses(True)
    
    game.set_screen_resolution(vzd.ScreenResolution.RES_320X200)

    # Set cv2 friendly format.
    game.set_screen_format(vzd.ScreenFormat.RGB24)

    # Enables labeling of the in game objects.
    game.set_labels_buffer_enabled(True)

    game.clear_available_game_variables()
    game.add_available_game_variable(vzd.GameVariable.POSITION_X)
    game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
    game.add_available_game_variable(vzd.GameVariable.POSITION_Z)

    game.set_available_buttons([vzd.Button.MOVE_LEFT,vzd.Button.MOVE_RIGHT ,vzd.Button.TURN_LEFT,vzd.Button.TURN_RIGHT ,vzd.Button.ATTACK ,vzd.Button.MOVE_FORWARD ,vzd.Button.MOVE_BACKWARD ,vzd.Button.USE])
    game.set_window_visible(True)
    game.init()
    episodes = 10000

    # Sleep time between actions in ms
    sleep_time = 0#1.0 / vzd.DEFAULT_TICRATE

    for i in range(episodes):
        print("Episode #" + str(i + 1))
        seen_in_this_episode = set()

        # Not needed for the first episode but the loop is nicer.
        game.new_episode()
        count = 0
        while not game.is_episode_finished():

            # Get the state
            state = game.get_state()
            game.advance_action()

            # Get labels buffer, that is always in 8-bit grey channel format.
            # Show only visible game objects (enemies, pickups, exploding barrels etc.), each with a unique label.
            # Additional labels data are available in state.labels.
            
            #if labels is not None:
            #    cv2.imshow('ViZDoom Labels Buffer', color_labels(labels))

            # Get screen buffer, given in selected format. This buffer is always available.
            # Using information from state.labels draw bounding boxes.
            #screen = state.screen_buffer
            #for l in state.labels:
            #    if l.object_name in ["Medkit", "GreenArmor"]:
            #        draw_bounding_box(screen, l.x, l.y, l.width, l.height, doom_blue_color)
            #    else:
            #        draw_bounding_box(screen, l.x, l.y, l.width, l.height, doom_red_color)
            #cv2.imshow('ViZDoom Screen Buffer', screen)
            #cv2.waitKey(sleep_time)

            # Make random action
            #game.make_action(choice(actions))

            #print("State #" + str(state.number))
            #print("Player position: x:", state.game_variables[0], ", y:", state.game_variables[1], ", z:", state.game_variables[2])
            #print("Labels:")

            # Print information about objects visible on the screen.
            # object_id identifies a specific in-game object.
            # It's unique for each object instance (two objects of the same type on the screen will have two different ids).
            # object_name contains the name of the object (can be understood as type of the object).
            # value tells which value represents the object in labels_buffer.
            # Values decrease with the distance from the player.
            # Objects with higher values (closer ones) can obscure objects with lower values (further ones).
            
            if count % 35 == 0:
              labels = state.labels_buffer
              for l in state.labels:
                  seen_in_this_episode.add(l.object_name)
                  #print("---------------------")
                  print("Label:", l.value, ", object id:", l.object_id, ", object name:", l.object_name)
                  #print("Object position: x:", l.object_position_x, ", y:", l.object_position_y, ", z:", l.object_position_z)

                  # Other available fields (position and velocity and bounding box):
                  #print("Object rotation angle", l.object_angle, "pitch:", l.object_pitch, "roll:", l.object_roll)
                  #print("Object velocity x:", l.object_velocity_x, "y:", l.object_velocity_y, "z:", l.object_velocity_z)
                  #print("Bounding box: x:", l.x, ", y:", l.y, ", width:", l.width, ", height:", l.height)
            count +=1


        print("Episode finished!")

        print("=====================")

        print("Unique objects types seen in this episode:")
        for l in seen_in_this_episode:
            print(l)

        print("************************")

    #cv2.destroyAllWindows()
    game.close()